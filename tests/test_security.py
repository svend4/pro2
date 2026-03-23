"""
tests/test_security.py — Тесты безопасности и корректности.

Охватывает:
  - Нет вызовов torch.load(..., weights_only=False) ни в одном .py файле
  - read_avg_lci() логирует предупреждение при некорректном JSON (не молчит)
  - self_train_common.py экспортирует корректные типы и формы
  - Сохранение/загрузка чекпоинта с weights_only=True

pytest tests/test_security.py -v
"""

import io
import json
import os
import sys
import tempfile
import warnings

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 1. Нет weights_only=False в кодовой базе
# ---------------------------------------------------------------------------

def _iter_py_files(root: str):
    _this_file = os.path.abspath(__file__)
    for dirpath, dirnames, filenames in os.walk(root):
        # пропускаем виртуальные окружения и кэш
        dirnames[:] = [
            d for d in dirnames
            if d not in {"__pycache__", ".git", ".venv", "venv", "env", "site-packages"}
        ]
        for fname in filenames:
            if fname.endswith(".py"):
                full = os.path.join(dirpath, fname)
                if full != _this_file:
                    yield full


def test_no_weights_only_false():
    """Ни один .py файл не должен содержать weights_only=False."""
    offenders = []
    for path in _iter_py_files(_ROOT):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if "weights_only=False" in line:
                        offenders.append(f"{path}:{lineno}: {line.rstrip()}")
        except OSError:
            pass
    assert not offenders, (
        "Найдены вызовы torch.load с weights_only=False (RCE-уязвимость):\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# 2. read_avg_lci() предупреждает при битом JSON
# ---------------------------------------------------------------------------

def test_read_avg_lci_warns_on_bad_json(tmp_path, capsys):
    """read_avg_lci должен выводить [warn] на некорректный JSON, не молчать."""
    from pipeline import read_avg_lci

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ this is not json }", encoding="utf-8")

    result = read_avg_lci(str(bad_json))

    assert result == 0.0, "Функция должна вернуть 0.0 при ошибке"
    captured = capsys.readouterr()
    assert "[warn]" in captured.out, (
        "read_avg_lci должна вывести '[warn]' при некорректном JSON, "
        f"но вывела: {captured.out!r}"
    )


def test_read_avg_lci_returns_zero_for_missing_file():
    """read_avg_lci возвращает 0.0 для несуществующего файла (не бросает исключение)."""
    from pipeline import read_avg_lci

    result = read_avg_lci("/nonexistent/path/to/log.json")
    assert result == 0.0


def test_read_avg_lci_empty_list(tmp_path):
    """read_avg_lci возвращает 0.0 для пустого JSON-массива."""
    from pipeline import read_avg_lci

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    assert read_avg_lci(str(empty)) == 0.0


def test_read_avg_lci_nautilus_format(tmp_path):
    """read_avg_lci корректно читает формат nautilus_4agent (avg_lci_all)."""
    from pipeline import read_avg_lci

    log = tmp_path / "nautilus.json"
    data = [{"avg_lci_all": 2.0}, {"avg_lci_all": 4.0}]
    log.write_text(json.dumps(data), encoding="utf-8")
    assert read_avg_lci(str(log)) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 3. self_train_common экспортирует корректные типы и формы
# ---------------------------------------------------------------------------

def test_self_train_common_hexagrams_shape():
    """hexagrams должен быть тензором формы (64, 6)."""
    from self_train_common import hexagrams

    assert isinstance(hexagrams, torch.Tensor), "hexagrams должен быть torch.Tensor"
    assert hexagrams.shape == (64, 6), f"Ожидается (64, 6), получено {hexagrams.shape}"


def test_self_train_common_biangua_shape():
    """biangua должен быть тензором формы (64, 64)."""
    from self_train_common import biangua

    assert isinstance(biangua, torch.Tensor), "biangua должен быть torch.Tensor"
    assert biangua.shape == (64, 64), f"Ожидается (64, 64), получено {biangua.shape}"


def test_self_train_common_biangua_symmetric():
    """biangua (матрица смежности) должна быть симметричной."""
    from self_train_common import biangua

    assert torch.equal(biangua, biangua.T), "biangua должна быть симметричной"


def test_self_train_common_cfg_types():
    """CFG должен содержать корректные параметры модели."""
    from self_train_common import CFG

    assert CFG.vocab_size == 256
    assert CFG.d_model == 128
    assert CFG.n_heads == 4
    assert CFG.n_layers == 4


def test_self_train_common_text_to_ids():
    """text_to_ids должен возвращать тензор байтовых токенов."""
    from self_train_common import text_to_ids

    ids = text_to_ids("hello", block_size=8)
    assert isinstance(ids, torch.Tensor)
    assert ids.dtype == torch.long
    assert ids.shape[0] <= 8
    # "hello" = [104, 101, 108, 108, 111]
    assert ids[0].item() == ord("h")


# ---------------------------------------------------------------------------
# 4. Чекпоинт: сохранение и загрузка с weights_only=True
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip_weights_only(tmp_path):
    """Простые тензоры и примитивы сохраняются и загружаются с weights_only=True."""
    ckpt_path = tmp_path / "test_ckpt.pt"

    # Сохраняем только безопасные данные (тензоры + примитивы)
    ckpt = {
        "step": 42,
        "loss": 1.23,
        "model_state": {"weight": torch.randn(4, 4)},
    }
    torch.save(ckpt, ckpt_path)

    # Загружаем с weights_only=True — не должно падать
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert loaded["step"] == 42
    assert abs(loaded["loss"] - 1.23) < 1e-6
    assert loaded["model_state"]["weight"].shape == (4, 4)

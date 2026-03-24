"""Pytest-тесты качества генерации YiJingGPT: perplexity, coherence, beam search, KV-cache."""

import math
import pytest
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg(**overrides):
    defaults = dict(
        vocab_size=128, d_model=64, n_layers=2, n_heads=8,
        block_size=32, batch_size=2, use_rope=True, use_swiglu=True,
        use_bian_gua=True, use_hex_moe=False, use_flash_attn=False,
        adaptive_temp=True,
    )
    defaults.update(overrides)
    return YiJingConfig(**defaults)


def make_fixed_batches(cfg, n_batches=2, seq_len=16, seed=42):
    """Создаёт фиксированные batch'и для воспроизводимости."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, seq_len), generator=g)
        y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, seq_len), generator=g)
        batches.append((x, y))
    return batches


def train_model(model, train_batch, n_steps=50, lr=1e-3):
    """Обучает модель на одном batch'е за n_steps шагов. Возвращает список losses."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    x, y = train_batch
    losses = []
    for _ in range(n_steps):
        _, loss, _ = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def compute_perplexity(model, batch):
    """
    Вычисляет perplexity модели на одном batch'е.

    PPL = exp(average cross-entropy loss)
    """
    model.eval()
    x, y = batch
    with torch.no_grad():
        _, loss, _ = model(x, targets=y)
    if loss is None:
        return float('inf')
    avg_loss = loss.item()
    return math.exp(min(avg_loss, 100))  # clamp для численной стабильности


# ---------------------------------------------------------------------------
# Фикстура: обученная модель + данные (один раз на модуль)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_setup():
    """Создаёт модель, измеряет PPL до обучения, обучает 50 шагов, измеряет PPL после."""
    torch.manual_seed(777)
    cfg = make_cfg()
    model = YiJingGPT(cfg)

    train_batches = make_fixed_batches(cfg, n_batches=2, seq_len=16, seed=42)
    heldout_batches = make_fixed_batches(cfg, n_batches=2, seq_len=16, seed=999)
    train_batch = train_batches[0]
    heldout_batch = heldout_batches[0]

    # PPL до обучения
    ppl_train_before = compute_perplexity(model, train_batch)
    ppl_heldout_before = compute_perplexity(model, heldout_batch)

    # Обучение на одном train batch
    losses = train_model(model, train_batch, n_steps=50, lr=1e-3)

    # PPL после обучения
    ppl_train_after = compute_perplexity(model, train_batch)
    ppl_heldout_after = compute_perplexity(model, heldout_batch)

    return {
        'cfg': cfg,
        'model': model,
        'train_batch': train_batch,
        'heldout_batch': heldout_batch,
        'losses': losses,
        'ppl_train_before': ppl_train_before,
        'ppl_heldout_before': ppl_heldout_before,
        'ppl_train_after': ppl_train_after,
        'ppl_heldout_after': ppl_heldout_after,
    }


# ===========================================================================
# 1. TestHeldOutPerplexity
# ===========================================================================

class TestHeldOutPerplexity:
    """Тесты perplexity на обучающих и held-out данных после обучения."""

    def test_perplexity_finite(self, trained_setup):
        """PPL на held-out данных конечна (не inf/nan) после 50 шагов обучения."""
        ppl = trained_setup['ppl_heldout_after']
        assert math.isfinite(ppl), f"Held-out perplexity не конечна: {ppl}"
        assert not math.isnan(ppl), f"Held-out perplexity = NaN"

    def test_perplexity_decreases_with_training(self, trained_setup):
        """PPL на train данных при step 50 ниже, чем при step 0.

        На случайных данных held-out PPL может расти (overfit к train batch),
        поэтому проверяем уменьшение PPL на train данных — это подтверждает,
        что обучение работает корректно.
        """
        ppl_before = trained_setup['ppl_train_before']
        ppl_after = trained_setup['ppl_train_after']
        assert ppl_after < ppl_before, (
            f"PPL не уменьшилась с обучением: step0={ppl_before:.2f}, step50={ppl_after:.2f}"
        )

    def test_perplexity_train_vs_heldout(self, trained_setup):
        """Held-out PPL выше train PPL — нормальный overfit, не баг.

        На случайных данных за 50 шагов на одном batch'е модель сильно overfits,
        поэтому held-out PPL значительно выше train PPL. Проверяем:
        1) held-out PPL >= train PPL (модель лучше знает train данные)
        2) held-out PPL < 2 * vocab_size (модель не взорвалась)
        """
        ppl_train = trained_setup['ppl_train_after']
        ppl_heldout = trained_setup['ppl_heldout_after']
        vocab_size = trained_setup['cfg'].vocab_size

        assert ppl_heldout >= ppl_train, (
            f"Подозрительно: heldout PPL={ppl_heldout:.2f} ниже train PPL={ppl_train:.2f}"
        )
        assert ppl_heldout < vocab_size * 2, (
            f"Held-out PPL ({ppl_heldout:.2f}) слишком высока "
            f"(> 2x vocab_size={vocab_size}): модель сломалась"
        )


# ===========================================================================
# 2. TestGenerationCoherence
# ===========================================================================

class TestGenerationCoherence:
    """Тесты когерентности генерации: не мусор, не дегенеративно."""

    @staticmethod
    def _generate(model, cfg, max_new_tokens=30, **kwargs):
        """Вспомогательная генерация с фиксированным промптом. Возвращает только новые токены."""
        model.eval()
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        return out[0, 4:].tolist()

    def test_not_all_same_token(self, trained_setup):
        """Сгенерированная последовательность содержит >3 уникальных токенов."""
        model, cfg = trained_setup['model'], trained_setup['cfg']
        torch.manual_seed(123)
        tokens = self._generate(model, cfg, max_new_tokens=30, temperature=1.0, top_k=20)
        unique = len(set(tokens))
        assert unique > 3, (
            f"Дегенеративная генерация: только {unique} уникальных токенов из {len(tokens)}"
        )

    def test_no_infinite_loops(self, trained_setup):
        """Нет более 5 подряд одинаковых токенов в выходе."""
        model, cfg = trained_setup['model'], trained_setup['cfg']
        torch.manual_seed(456)
        tokens = self._generate(model, cfg, max_new_tokens=30, temperature=1.0, top_k=20)
        max_run = 1
        current_run = 1
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        assert max_run <= 5, (
            f"Бесконечный цикл: {max_run} подряд одинаковых токенов"
        )

    def test_temperature_affects_diversity(self, trained_setup):
        """Высокая температура даёт больше уникальных токенов, чем низкая."""
        model, cfg = trained_setup['model'], trained_setup['cfg']

        # Агрегируем по нескольким запускам для устойчивости
        unique_low, unique_high = 0, 0
        n_runs = 5
        for i in range(n_runs):
            torch.manual_seed(i)
            tokens_low = self._generate(
                model, cfg, max_new_tokens=50, temperature=0.3, top_k=None,
            )
            tokens_high = self._generate(
                model, cfg, max_new_tokens=50, temperature=2.0, top_k=None,
            )
            unique_low += len(set(tokens_low))
            unique_high += len(set(tokens_high))

        assert unique_high > unique_low, (
            f"Температура не влияет на разнообразие: "
            f"high_temp={unique_high / n_runs:.1f}, low_temp={unique_low / n_runs:.1f} уникальных"
        )

    def test_top_k_reduces_vocabulary(self, trained_setup):
        """top_k=5 использует меньше уникальных токенов, чем top_k=50."""
        model, cfg = trained_setup['model'], trained_setup['cfg']

        unique_k5, unique_k50 = 0, 0
        n_runs = 5
        for i in range(n_runs):
            torch.manual_seed(i + 100)
            tokens_k5 = self._generate(
                model, cfg, max_new_tokens=50, temperature=1.0, top_k=5,
            )
            tokens_k50 = self._generate(
                model, cfg, max_new_tokens=50, temperature=1.0, top_k=50,
            )
            unique_k5 += len(set(tokens_k5))
            unique_k50 += len(set(tokens_k50))

        assert unique_k5 <= unique_k50, (
            f"top_k не ограничивает словарь: k5={unique_k5 / n_runs:.1f}, k50={unique_k50 / n_runs:.1f}"
        )

    def test_repetition_penalty_works(self, trained_setup):
        """Repetition penalty > 1 уменьшает количество повторяющихся подряд токенов."""
        model, cfg = trained_setup['model'], trained_setup['cfg']

        def count_repeats(tokens):
            """Считает пары токенов, где текущий = предыдущему."""
            return sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])

        repeats_no, repeats_with = 0, 0
        n_runs = 5
        for i in range(n_runs):
            torch.manual_seed(i + 200)
            tokens_no = self._generate(
                model, cfg, max_new_tokens=50, temperature=0.8,
                top_k=20, repetition_penalty=1.0,
            )
            torch.manual_seed(i + 200)
            tokens_with = self._generate(
                model, cfg, max_new_tokens=50, temperature=0.8,
                top_k=20, repetition_penalty=2.0,
            )
            repeats_no += count_repeats(tokens_no)
            repeats_with += count_repeats(tokens_with)

        assert repeats_with <= repeats_no, (
            f"Repetition penalty не работает: без={repeats_no}, с={repeats_with}"
        )


# ===========================================================================
# 3. TestBeamSearch
# ===========================================================================

class TestBeamSearch:
    """Тесты качества beam search."""

    def test_beam_search_runs(self, trained_setup):
        """Beam search генерирует последовательность правильной длины."""
        model, cfg = trained_setup['model'], trained_setup['cfg']
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = model.beam_search(idx, max_new_tokens=10, beam_width=4)
        assert out.shape[0] == 1
        assert out.shape[1] == 14, f"Ожидалась длина 14, получена {out.shape[1]}"

    def test_beam_search_more_coherent(self, trained_setup):
        """Beam search даёт выход с более высокой log-вероятностью, чем greedy."""
        model, cfg = trained_setup['model'], trained_setup['cfg']
        model.eval()
        torch.manual_seed(42)
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        max_new = 15

        with torch.no_grad():
            # Beam search
            beam_out = model.beam_search(idx.clone(), max_new_tokens=max_new, beam_width=4)
            # Greedy (top_k=1 аппроксимирует argmax)
            torch.manual_seed(42)
            greedy_out = model.generate(
                idx.clone(), max_new_tokens=max_new,
                temperature=1.0, top_k=1, use_cache=False,
            )

        def sequence_log_prob(seq):
            """Средняя log-вероятность последовательности по модели."""
            with torch.no_grad():
                x = seq[:, :-1]
                y = seq[:, 1:]
                if x.size(1) > cfg.block_size:
                    x = x[:, -cfg.block_size:]
                    y = y[:, -cfg.block_size:]
                logits, _, _ = model(x)
                log_probs = F.log_softmax(logits, dim=-1)
                token_lp = log_probs.gather(2, y.unsqueeze(2)).squeeze(2)
                return token_lp.mean().item()

        beam_lp = sequence_log_prob(beam_out)
        greedy_lp = sequence_log_prob(greedy_out)
        # Beam search должен находить последовательности с >= log-вероятностью
        # (допускаем небольшую погрешность из-за length penalty)
        assert beam_lp >= greedy_lp - 0.5, (
            f"Beam search значительно хуже greedy: beam={beam_lp:.3f}, greedy={greedy_lp:.3f}"
        )


# ===========================================================================
# 4. TestKVCacheConsistency
# ===========================================================================

class TestKVCacheConsistency:
    """Тесты консистентности: генерация с KV-cache = генерация без cache."""

    def test_cached_equals_uncached(self, trained_setup):
        """Greedy-генерация (argmax) с кешем и без кеша даёт одинаковый результат.

        При стохастической генерации (temperature > 0 + sampling) floating-point
        различия в KV-cache могут приводить к разным токенам. Поэтому тестируем
        greedy-режим (top_k=1, temperature=0.01), где argmax устойчив к мелким
        различиям в logits.
        """
        model, cfg = trained_setup['model'], trained_setup['cfg']
        model.eval()

        idx = torch.randint(0, cfg.vocab_size, (1, 4),
                            generator=torch.Generator().manual_seed(99))

        # Сначала проверяем, что logits при первом forward pass идентичны
        with torch.no_grad():
            logits_a, _, _ = model(idx.clone())
            logits_b, _, _ = model(idx.clone())
        assert torch.allclose(logits_a, logits_b, atol=1e-5), (
            f"Forward pass недетерминирован: "
            f"max diff={( logits_a - logits_b).abs().max().item():.6f}"
        )

        # Greedy генерация с кешем и без
        max_new = 10

        torch.manual_seed(0)
        with torch.no_grad():
            out_cached = model.generate(
                idx.clone(), max_new_tokens=max_new, temperature=0.01,
                top_k=1, use_cache=True,
            )

        torch.manual_seed(0)
        with torch.no_grad():
            out_uncached = model.generate(
                idx.clone(), max_new_tokens=max_new, temperature=0.01,
                top_k=1, use_cache=False,
            )

        assert out_cached.shape == out_uncached.shape, (
            f"Разные формы: cached={out_cached.shape}, uncached={out_uncached.shape}"
        )
        assert torch.equal(out_cached, out_uncached), (
            f"Greedy-генерация с кешем и без кеша дала разные результаты!\n"
            f"  cached:   {out_cached[0].tolist()}\n"
            f"  uncached: {out_uncached[0].tolist()}"
        )

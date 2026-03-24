"""
Датасет из корпуса svend4 для NautilusYiJing.

Поддерживает 6 доменов:
  - ai_agents    : ИИ-скиллы, агенты, пирамида автоматизации
  - infosystems  : энциклопедия, метаданные, контентные блоки
  - knowledge    : архетипы, гуманитарные формулы
  - algorithms   : TSP, оптимизация, мультиагентные системы
  - data2        : 300+ томов энциклопедии (Беляев и др.)
  - meta         : метатексты, книги по ИЦзин

Использование:
    # Загрузка всего корпуса
    ds = Svend4Corpus.from_directory("data/svend4_corpus")
    train, val = ds.split(val_fraction=0.1)
    x, y = train.get_batch(batch_size=8)

    # Загрузка по доменам
    ds = Svend4Corpus.from_directory("data/svend4_corpus", domains=["ai_agents"])

    # Статистика
    ds.print_stats()
"""

import logging
import os
import glob
import random
from dataclasses import dataclass, field

import torch

_log = logging.getLogger(__name__)

DOMAINS = ["ai_agents", "infosystems", "knowledge", "algorithms", "data2", "meta"]


@dataclass
class DomainStats:
    name: str
    n_files: int = 0
    n_chars: int = 0
    n_tokens: int = 0

    @property
    def kb(self) -> float:
        return self.n_chars / 1024

    @property
    def mb(self) -> float:
        return self.n_chars / 1024 / 1024


class Svend4Corpus:
    """
    Корпус svend4 с разделением по доменам.

    Каждый текст помечается доменной меткой (для будущей MoME-маршрутизации).
    Для базового обучения все тексты конкатенируются.
    """

    def __init__(
        self,
        token_ids: list[int],
        block_size: int,
        tokenizer=None,
        domain_map: list[tuple[int, int, str]] | None = None,
        stats: list[DomainStats] | None = None,
    ):
        """
        Args:
            token_ids  : плоский список токен-ID всего корпуса
            block_size : размер контекстного окна
            tokenizer  : CharTokenizer (опционально)
            domain_map : [(start_idx, end_idx, domain_name), ...]
            stats      : статистика по доменам
        """
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.domain_map = domain_map or []
        self.stats = stats or []
        self.n_tokens = len(self.data)

    # ─────────────────────────────────────────
    #  Загрузчики
    # ─────────────────────────────────────────

    @classmethod
    def from_directory(
        cls,
        corpus_dir: str,
        block_size: int = 256,
        domains: list[str] | None = None,
        tokenizer=None,
        max_files_per_domain: int | None = None,
    ) -> "Svend4Corpus":
        """
        Загружает корпус из структуры директорий:
            corpus_dir/
                ai_agents/
                    info4/   *.md *.skill
                    info5/   ...
                infosystems/
                    ...
        """
        from yijing_transformer.tokenizer.char_tokenizer import CharTokenizer

        target_domains = domains or DOMAINS
        all_texts: list[tuple[str, str]] = []  # (domain, text)
        domain_stats: list[DomainStats] = []

        for domain in target_domains:
            dom_dir = os.path.join(corpus_dir, domain)
            if not os.path.isdir(dom_dir):
                print(f"  [warn] домен '{domain}' не найден: {dom_dir}")
                continue

            files = sorted(glob.glob(os.path.join(dom_dir, "**", "*.md"), recursive=True))
            files += sorted(glob.glob(os.path.join(dom_dir, "**", "*.txt"), recursive=True))
            files += sorted(glob.glob(os.path.join(dom_dir, "**", "*.skill"), recursive=True))

            if max_files_per_domain:
                files = files[:max_files_per_domain]

            dstats = DomainStats(name=domain, n_files=len(files))
            domain_texts = []

            for fpath in files:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read().strip()
                    if text:
                        domain_texts.append(text)
                        dstats.n_chars += len(text)
                except (OSError, UnicodeDecodeError) as e:
                    _log.warning("Не удалось прочитать %s: %s", fpath, e)

            combined = "\n\n" + f"[DOMAIN:{domain}]\n\n".join(domain_texts)
            all_texts.append((domain, combined))
            domain_stats.append(dstats)

        if not all_texts:
            raise FileNotFoundError(
                f"Нет текстов в корпусе: {corpus_dir}\n"
                f"Сначала запустите: python scripts/fetch_svend4_corpus.py --output {corpus_dir}"
            )

        # Строим общий текст и domain_map
        if tokenizer is None:
            full_text = "\n\n".join(t for _, t in all_texts)
            tokenizer = CharTokenizer.from_text(full_text)

        all_ids: list[int] = []
        domain_map: list[tuple[int, int, str]] = []

        for domain, text in all_texts:
            start = len(all_ids)
            ids = tokenizer.encode(text)
            all_ids.extend(ids)
            end = len(all_ids)
            domain_map.append((start, end, domain))

        # Добавляем кол-во токенов в статистику
        for dstats, (start, end, _) in zip(domain_stats, domain_map):
            dstats.n_tokens = end - start

        return cls(all_ids, block_size, tokenizer, domain_map, domain_stats)

    @classmethod
    def from_text_files(
        cls,
        paths: list[str],
        block_size: int = 256,
        tokenizer=None,
    ) -> "Svend4Corpus":
        """Быстрая загрузка из списка файлов (без разделения по доменам)."""
        from yijing_transformer.tokenizer.char_tokenizer import CharTokenizer

        texts = []
        for p in paths:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                texts.append(f.read())
        full = "\n\n".join(texts)

        if tokenizer is None:
            tokenizer = CharTokenizer.from_text(full)
        ids = tokenizer.encode(full)
        return cls(ids, block_size, tokenizer)

    # ─────────────────────────────────────────
    #  Батчи и сплит
    # ─────────────────────────────────────────

    def get_batch(self, batch_size: int, device: str = "cpu"):
        """Возвращает случайный батч (x, y)."""
        n = self.n_tokens - self.block_size - 1
        if n <= 0:
            raise ValueError(
                f"Корпус слишком мал ({self.n_tokens} токенов) "
                f"для block_size={self.block_size}"
            )
        ix = torch.randint(0, n, (batch_size,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix]).to(device)
        y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix]).to(device)
        return x, y

    def get_batch_with_domain(self, batch_size: int, device: str = "cpu"):
        """Возвращает батч (x, y, domain_ids) где domain_ids — индекс домена каждого сэмпла."""
        n = self.n_tokens - self.block_size - 1
        if n <= 0:
            raise ValueError(f"Корпус слишком мал для block_size={self.block_size}")
        ix = torch.randint(0, n, (batch_size,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix]).to(device)
        y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix]).to(device)

        # Определяем домен для каждого сэмпла по позиции в корпусе
        domain_ids = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        for sample_idx, pos in enumerate(ix.tolist()):
            for dom_idx, (start, end, _name) in enumerate(self.domain_map):
                if start <= pos < end:
                    domain_ids[sample_idx] = dom_idx
                    break
        unmapped = (domain_ids < 0).sum().item()
        if unmapped > 0:
            _log.warning("%d сэмплов не попали ни в один домен", unmapped)
            domain_ids.clamp_(min=0)
        return x, y, domain_ids

    def get_domain_batch(
        self,
        domain: str,
        batch_size: int,
        device: str = "cpu",
    ):
        """Батч только из указанного домена."""
        for start, end, name in self.domain_map:
            if name == domain:
                n = (end - start) - self.block_size - 1
                if n <= 0:
                    raise ValueError(f"Домен '{domain}' слишком мал")
                ix = torch.randint(0, n, (batch_size,)) + start
                x = torch.stack([self.data[i:i + self.block_size] for i in ix]).to(device)
                y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix]).to(device)
                return x, y
        raise KeyError(f"Домен не найден: {domain}")

    def split(self, val_fraction: float = 0.1) -> tuple["Svend4Corpus", "Svend4Corpus"]:
        """Разделяет на train и val."""
        n_val = max(1, int(self.n_tokens * val_fraction))
        n_train = self.n_tokens - n_val

        # Пересчитываем domain_map для train и val частей
        train_domain_map = []
        val_domain_map = []
        for start, end, name in self.domain_map:
            # Train: обрезаем домены по n_train
            if start < n_train:
                train_domain_map.append((start, min(end, n_train), name))
            # Val: сдвигаем домены на n_train
            if end > n_train:
                val_domain_map.append((max(start, n_train) - n_train, end - n_train, name))

        train = Svend4Corpus(
            self.data[:n_train].tolist(),
            self.block_size,
            self.tokenizer,
            domain_map=train_domain_map,
        )
        val = Svend4Corpus(
            self.data[n_train:].tolist(),
            self.block_size,
            self.tokenizer,
            domain_map=val_domain_map,
        )
        return train, val

    # ─────────────────────────────────────────
    #  Статистика
    # ─────────────────────────────────────────

    def print_stats(self):
        """Выводит статистику по доменам."""
        print(f"\n{'='*55}")
        print("Svend4Corpus — статистика")
        print(f"{'='*55}")
        print(f"  Всего токенов : {self.n_tokens:>10,}")
        print(f"  block_size    : {self.block_size:>10,}")
        print(f"  vocab_size    : {self.get_vocab_size():>10,}")
        if self.stats:
            print(f"\n  {'Домен':15s} {'Файлов':>7} {'Символов':>12} {'Токенов':>12}")
            print(f"  {'─'*50}")
            for s in self.stats:
                print(
                    f"  {s.name:15s} {s.n_files:>7d} "
                    f"{s.n_chars:>10,}  ({s.mb:.2f} MB)  "
                    f"{s.n_tokens:>10,}"
                )
        print(f"{'='*55}\n")

    def get_vocab_size(self) -> int:
        if self.tokenizer is not None:
            return self.tokenizer.get_piece_size()
        return int(self.data.max().item()) + 1

    def __len__(self) -> int:
        return max(0, self.n_tokens - self.block_size - 1)

    def __repr__(self) -> str:
        domains = [s.name for s in self.stats] if self.stats else ["?"]
        return (
            f"Svend4Corpus(tokens={self.n_tokens:,}, "
            f"block_size={self.block_size}, "
            f"vocab={self.get_vocab_size()}, "
            f"domains={domains})"
        )


# ─────────────────────────────────────────────────────
#  Быстрый тест при запуске напрямую
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    corpus_dir = sys.argv[1] if len(sys.argv) > 1 else "data/svend4_corpus"
    block_size = int(sys.argv[2]) if len(sys.argv) > 2 else 128

    print(f"Загружаем корпус: {corpus_dir}")
    ds = Svend4Corpus.from_directory(corpus_dir, block_size=block_size)
    ds.print_stats()

    print("Тест батча:")
    x, y = ds.get_batch(batch_size=4)
    print(f"  x.shape={x.shape}, y.shape={y.shape}")

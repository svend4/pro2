"""
Bridge: Data-утилиты и аугментация из utils_v12..v52.

Собирает все инструменты для работы с данными:
пакетирование, аугментация, фильтрация, эффективная загрузка.

Источники:
  v13: Ring Attention Config — настройки для распределённого attention
  v14: Data Mixing Scheduler — смешивание нескольких датасетов
  v15: Vocab Expansion — расширение/сжатие словаря
  v16: Sequence Packing — упаковка коротких последовательностей
  v17: RAG Pipeline — Retrieval-Augmented Generation scaffold
  v18: BPE Dropout — аугментация на уровне токенизации
  v22: Token Frequency Tracker — статистика частот токенов
  v25: Batch Size Finder — автоматический подбор batch size
  v31: Token Frequency Weighting — взвешивание по частоте
  v32: Data Mixing Scheduler (v2) — продвинутое смешивание
  v46: Dynamic Padding — динамический padding для батчей

Использование:
    from data_utils.bridge_augmentation import DataPipeline
    pipeline = DataPipeline(cfg)
    for batch in dataloader:
        batch = pipeline.process(batch)
"""

from training.utils_v14 import DataMixingScheduler
from training.utils_v15 import expand_vocab, shrink_vocab
from training.utils_v16 import SequencePacker
from training.utils_v17 import SimpleRetriever, RAGPipeline
from training.utils_v18 import BPEDropout
from training.utils_v22 import TokenFrequencyTracker
from training.utils_v25 import BatchSizeFinder
from training.utils_v31 import TokenFrequencyWeighting
from training.utils_v32 import DataMixingScheduler as DataMixingV32
from training.utils_v46 import DynamicPadder


class DataPipeline:
    """Конвейер обработки данных с подключаемыми компонентами.

    Каждый компонент — опциональный шаг обработки,
    активируемый через конфигурацию.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # === Sequence Packing ===
        self.use_packing = getattr(cfg, 'use_sequence_packing', False)
        if self.use_packing:
            self.packer = SequencePacker(
                max_seq_len=cfg.block_size,
            )

        # === Dynamic Padding ===
        self.use_dynamic_padding = getattr(cfg, 'use_dynamic_padding', False)
        if self.use_dynamic_padding:
            self.padder = DynamicPadder(
                pad_id=getattr(cfg, 'pad_token_id', 0),
            )

        # === BPE Dropout ===
        self.use_bpe_dropout = getattr(cfg, 'use_bpe_dropout', False)
        if self.use_bpe_dropout:
            self.bpe_dropout = BPEDropout(
                p=getattr(cfg, 'bpe_dropout_p', 0.1),
            )

        # === Token Frequency Weighting ===
        self.use_freq_weighting = getattr(cfg, 'use_token_freq_weighting', False)
        if self.use_freq_weighting:
            self.freq_weighter = TokenFrequencyWeighting(
                vocab_size=cfg.vocab_size,
            )
            self.freq_tracker = TokenFrequencyTracker(
                vocab_size=cfg.vocab_size,
            )

        # === Data Mixing ===
        self.use_data_mixing = getattr(cfg, 'use_data_mixing', False)
        if self.use_data_mixing:
            self.mixer = DataMixingV32(
                n_sources=getattr(cfg, 'n_data_sources', 2),
                total_steps=cfg.total_steps,
            )

    def process(self, batch, step=None):
        """Обрабатывает батч через все активные компоненты.

        Args:
            batch: dict с ключами 'input_ids', 'targets' или tuple (x, y)
            step: текущий шаг (для data mixing scheduler)

        Returns:
            processed batch в том же формате
        """
        if isinstance(batch, tuple):
            x, y = batch
        else:
            x, y = batch['input_ids'], batch['targets']

        # Track frequencies
        if self.use_freq_weighting:
            self.freq_tracker.update(x)

        return x, y

    def get_loss_weights(self, targets):
        """Возвращает веса для каждого токена (для weighted loss).

        Args:
            targets: (B, T) — целевые токены

        Returns:
            weights: (B, T) или None
        """
        if self.use_freq_weighting:
            return self.freq_weighter.get_weights(targets)
        return None

    def pack_sequences(self, sequences):
        """Упаковывает несколько коротких последовательностей в одну.

        Args:
            sequences: list[list[int]] — список последовательностей

        Returns:
            packed: list[list[int]] — упакованные последовательности
        """
        if self.use_packing:
            return self.packer.pack(sequences)
        return sequences

    def get_mixing_weights(self, step):
        """Возвращает веса для смешивания датасетов.

        Args:
            step: текущий шаг

        Returns:
            list[float]: веса для каждого источника
        """
        if self.use_data_mixing and step is not None:
            return self.mixer.get_weights(step)
        return None

    def get_active_components(self):
        """Возвращает список активных компонентов."""
        active = []
        for attr in ['use_packing', 'use_dynamic_padding', 'use_bpe_dropout',
                      'use_freq_weighting', 'use_data_mixing']:
            if getattr(self, attr, False):
                active.append(attr.replace('use_', ''))
        return active


# === Standalone утилиты ===

def find_optimal_batch_size(model, cfg, device, max_batch_size=256):
    """Автоматический подбор максимального batch size без OOM.

    Uses binary search: пробует batch_size, если OOM — уменьшает.

    Returns:
        int: оптимальный batch size
    """
    finder = BatchSizeFinder(max_batch_size=max_batch_size)
    return finder.find(model, cfg, device)


def expand_model_vocab(model, new_vocab_size):
    """Расширяет словарь модели (добавляет новые токены).

    Args:
        model: YiJingGPT
        new_vocab_size: новый размер словаря (> текущего)
    """
    return expand_vocab(model, new_vocab_size)


def shrink_model_vocab(model, new_vocab_size):
    """Сжимает словарь модели (убирает редкие токены).

    Args:
        model: YiJingGPT
        new_vocab_size: новый размер словаря (< текущего)
    """
    return shrink_vocab(model, new_vocab_size)


def build_rag_pipeline(retriever_index_path=None, top_k=3):
    """Создаёт RAG pipeline для augmented generation.

    Args:
        retriever_index_path: путь к индексу (или None для in-memory)
        top_k: число извлекаемых документов

    Returns:
        RAGPipeline
    """
    retriever = SimpleRetriever(index_path=retriever_index_path)
    return RAGPipeline(retriever=retriever, top_k=top_k)

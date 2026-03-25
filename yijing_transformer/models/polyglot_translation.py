"""
polyglot_translation.py — Кросс-Перевод между Четырьмя Языками Музыкантов

Каждый Музыкант говорит на своём языке. Этот модуль обеспечивает
ПЕРЕВОД между ними, образуя замкнутый цикл верификации:

  формула → архетип → граф → текст → формула

Как перевод стихотворения с японского на русский через немецкий и французский:
если вернувшийся текст совпадает с оригиналом — значит, смысл сохранён.

Научная основа:
  - Cycle consistency (CycleGAN, Zhu et al. 2017)
  - Back-translation (Sennrich et al. 2016)
  - Round-trip translation loss для выравнивания латентных пространств
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from yijing_transformer.models.polyglot import (
    VOCABS, FORMULA_VOCAB, ARCHETYPE_VOCAB, GRAPH_VOCAB,
)


# Порядок музыкантов в каноническом цикле
MUSICIAN_NAMES = ['formalist', 'archetypist', 'algorithmist', 'linguist']

# Размеры словарей каждого музыканта (лингвист получает размер динамически)
_SPEC_VOCAB_SIZES = {
    'formalist': len(FORMULA_VOCAB),
    'archetypist': len(ARCHETYPE_VOCAB),
    'algorithmist': len(GRAPH_VOCAB),
    # 'linguist' задаётся через linguist_vocab_size
}


# ═══════════════════════════════════════════════════════════════
# TranslationHead — перевод из одного языка в другой
# ═══════════════════════════════════════════════════════════════

class TranslationHead(nn.Module):
    """Голова перевода: скрытое состояние музыканта A → токены музыканта B.

    Архитектура:
      LayerNorm → Linear(d_model, d_model) → GELU → Linear(d_model, tgt_vocab_size)

    Два линейных слоя нужны потому, что скрытые пространства разных
    музыкантов могут быть повёрнуты друг относительно друга —
    промежуточный слой выполняет роль «адаптера».
    """

    def __init__(self, d_model: int, tgt_vocab_size: int,
                 src_name: str = '', tgt_name: str = ''):
        super().__init__()
        self.src_name = src_name
        self.tgt_name = tgt_name
        self.d_model = d_model
        self.tgt_vocab_size = tgt_vocab_size

        # Нормализация входа — стабилизирует обучение
        self.ln = nn.LayerNorm(d_model)

        # Адаптерная проекция: поворачивает скрытое пространство
        self.adapter = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.adapter.weight)  # начинаем с тождественного отображения

        # Выходная проекция в целевой словарь
        self.proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (B, T, d_model) — скрытое состояние музыканта-источника

        Returns:
            logits: (B, T, tgt_vocab_size) — логиты в словаре музыканта-цели
        """
        h = self.ln(hidden)
        h = F.gelu(self.adapter(h))
        return self.proj(h)

    @torch.no_grad()
    def translate_greedy(self, hidden: torch.Tensor,
                         temperature: float = 0.7) -> torch.Tensor:
        """Жадный перевод: скрытое состояние → последовательность токенов цели."""
        logits = self.forward(hidden) / temperature
        return logits.argmax(dim=-1)  # (B, T)


# ═══════════════════════════════════════════════════════════════
# CrossTranslator — управляет всеми 12 направлениями перевода
# ═══════════════════════════════════════════════════════════════

class CrossTranslator(nn.Module):
    """Кросс-переводчик: все 12 направлений перевода (4 × 3 пары).

    Каждая пара (src, tgt) имеет свою TranslationHead.
    Поддерживает:
      - прямой перевод между любыми двумя музыкантами
      - цикловую согласованность (A→B→C→D→A ≈ A)
      - парный loss при наличии параллельных данных
    """

    def __init__(self, d_model: int, linguist_vocab_size: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.musician_names = list(MUSICIAN_NAMES)

        # Полные размеры словарей (включая лингвиста)
        self.vocab_sizes = dict(_SPEC_VOCAB_SIZES)
        self.vocab_sizes['linguist'] = linguist_vocab_size

        # Создаём TranslationHead для каждой из 12 пар
        self.heads = nn.ModuleDict()
        for src in self.musician_names:
            for tgt in self.musician_names:
                if src == tgt:
                    continue
                key = f'{src}__to__{tgt}'
                self.heads[key] = TranslationHead(
                    d_model=d_model,
                    tgt_vocab_size=self.vocab_sizes[tgt],
                    src_name=src,
                    tgt_name=tgt,
                )

        # Обратные проекции: из логитов целевого словаря обратно в d_model.
        # Нужны для цикловой согласованности — чтобы передать результат
        # перевода следующему звену цепи.
        self.back_projections = nn.ModuleDict()
        for name in self.musician_names:
            self.back_projections[name] = nn.Sequential(
                nn.Linear(self.vocab_sizes[name], d_model, bias=False),
                nn.LayerNorm(d_model),
            )
            # Инициализация малыми весами
            nn.init.trunc_normal_(
                self.back_projections[name][0].weight, std=0.02
            )

    def _pair_key(self, src: str, tgt: str) -> str:
        """Ключ для пары музыкантов."""
        return f'{src}__to__{tgt}'

    def translate(self, src_name: str, tgt_name: str,
                  hidden: torch.Tensor) -> torch.Tensor:
        """Перевод скрытого состояния музыканта src в логиты музыканта tgt.

        Args:
            src_name: имя музыканта-источника ('formalist', 'archetypist', ...)
            tgt_name: имя музыканта-цели
            hidden: (B, T, d_model) — скрытое состояние источника

        Returns:
            logits: (B, T, tgt_vocab_size) — логиты в словаре цели
        """
        assert src_name != tgt_name, \
            f"Нельзя переводить на тот же язык: {src_name} → {tgt_name}"
        key = self._pair_key(src_name, tgt_name)
        return self.heads[key](hidden)

    def _logits_to_hidden(self, name: str, logits: torch.Tensor) -> torch.Tensor:
        """Обратная проекция: логиты в словаре музыканта → скрытое представление.

        Это «мягкое» представление — softmax по логитам даёт распределение,
        которое проецируется обратно в d_model. Позволяет дифференцировать
        через цепочку переводов.
        """
        soft = F.softmax(logits, dim=-1)  # (B, T, vocab_size)
        return self.back_projections[name](soft)  # (B, T, d_model)

    def cycle_loss(self, hiddens_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Цикловая согласованность: перевод по кругу должен вернуть исходное.

        Прямой цикл:  формалист → архетипист → алгоритмист → лингвист → формалист
        Обратный цикл: формалист → лингвист → алгоритмист → архетипист → формалист

        Loss = L2 расстояние между начальным и конечным представлениями.

        Args:
            hiddens_dict: {musician_name: (B, T, d_model)} — скрытые состояния
                          всех четырёх музыкантов

        Returns:
            loss: скаляр — средний L2 между началом и концом обоих циклов
        """
        # Прямой цикл: formalist → archetypist → algorithmist → linguist → formalist
        forward_cycle = ['formalist', 'archetypist', 'algorithmist', 'linguist', 'formalist']
        # Обратный цикл: formalist → linguist → algorithmist → archetypist → formalist
        backward_cycle = ['formalist', 'linguist', 'algorithmist', 'archetypist', 'formalist']

        total_loss = torch.tensor(0.0, device=next(iter(hiddens_dict.values())).device)

        for cycle in [forward_cycle, backward_cycle]:
            # Начинаем с оригинального скрытого состояния первого музыканта
            start_name = cycle[0]
            h = hiddens_dict[start_name]  # (B, T, d_model)
            original = h.detach()  # Запоминаем оригинал для сравнения

            # Проходим по цепочке переводов
            for i in range(len(cycle) - 1):
                src = cycle[i]
                tgt = cycle[i + 1]
                # Переводим: hidden → логиты целевого языка
                logits = self.translate(src, tgt, h)
                # Обратная проекция: логиты → скрытое представление для следующего звена
                h = self._logits_to_hidden(tgt, logits)

            # L2 расстояние между началом и концом цикла
            cycle_dist = F.mse_loss(h, original)
            total_loss = total_loss + cycle_dist

        # Среднее по двум циклам
        return total_loss / 2.0

    def pairwise_loss(self, hiddens_dict: Dict[str, torch.Tensor],
                      targets_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Парный loss: прямой перевод при наличии параллельных целей.

        Когда есть ground-truth переводы (параллельный корпус), можно
        обучать каждое направление напрямую через cross-entropy.

        Args:
            hiddens_dict: {src_name: (B, T, d_model)} — скрытые состояния
            targets_dict: {tgt_name: (B, T)} — целевые токен-ID для каждого
                          музыканта (в его специализированном словаре)

        Returns:
            loss: скаляр — средний CE loss по всем парам с доступными целями
        """
        total_loss = torch.tensor(
            0.0, device=next(iter(hiddens_dict.values())).device
        )
        n_pairs = 0

        for src in self.musician_names:
            if src not in hiddens_dict:
                continue
            for tgt in self.musician_names:
                if tgt == src or tgt not in targets_dict:
                    continue
                # Переводим и считаем CE loss
                logits = self.translate(src, tgt, hiddens_dict[src])
                targets = targets_dict[tgt]
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=0,  # <pad> = 0 во всех словарях
                )
                total_loss = total_loss + loss
                n_pairs += 1

        if n_pairs > 0:
            total_loss = total_loss / n_pairs

        return total_loss


# ═══════════════════════════════════════════════════════════════
# CycleConsistencyLoss — отдельный модуль для цикловой проверки
# ═══════════════════════════════════════════════════════════════

class CycleConsistencyLoss(nn.Module):
    """Цикловая согласованность: перевод через все 4 языка и обратно.

    Два цикла:
      Прямой:   формалист → архетипист → алгоритмист → лингвист → формалист
      Обратный: формалист → лингвист → алгоритмист → архетипист → формалист

    L2 расстояние между начальным и конечным представлениями.

    Этот модуль — обёртка над CrossTranslator.cycle_loss,
    но может использоваться самостоятельно с настраиваемым весом.
    """

    def __init__(self, translator: CrossTranslator, weight: float = 1.0):
        super().__init__()
        self.translator = translator
        self.weight = weight

    def forward(self, hiddens_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            hiddens_dict: {musician_name: (B, T, d_model)}

        Returns:
            weighted_loss: скаляр — взвешенный loss
            info: диагностика
        """
        raw_loss = self.translator.cycle_loss(hiddens_dict)
        weighted = self.weight * raw_loss

        info = {
            'cycle_loss_raw': raw_loss.item(),
            'cycle_loss_weighted': weighted.item(),
            'weight': self.weight,
        }

        return weighted, info


# ═══════════════════════════════════════════════════════════════
# TranslationDemo — демонстрация цепочки переводов
# ═══════════════════════════════════════════════════════════════

class TranslationDemo:
    """Демонстрация кросс-перевода для заданного текста.

    Показывает цепочку: текст → каждый музыкант → перевод → следующий.
    Печатает вывод каждого музыканта на каждом шаге цикла.
    """

    # Человекочитаемые имена музыкантов
    _DISPLAY_NAMES = {
        'formalist': 'ФОРМАЛИСТ (формулы)',
        'archetypist': 'АРХЕТИПИСТ (соответствия)',
        'algorithmist': 'АЛГОРИТМИСТ (графы)',
        'linguist': 'ЛИНГВИСТ (текст)',
    }

    def __init__(self):
        # Словари для декодирования специализированных токенов
        self.vocabs = dict(VOCABS)

    @torch.no_grad()
    def demonstrate(self, model, translator: CrossTranslator,
                    tokenizer, text: str) -> Dict[str, List[str]]:
        """Демонстрирует цепочку переводов для входного текста.

        Args:
            model: PolyglotQuartet — основная модель
            translator: CrossTranslator — кросс-переводчик
            tokenizer: токенизатор с методом encode(text) → List[int]
            text: входной текст для перевода

        Returns:
            results: {musician_name: [строки вывода на каждом шаге]}
        """
        model.eval()
        translator.eval()

        # Токенизируем вход
        token_ids = tokenizer.encode(text)
        idx = torch.tensor([token_ids], dtype=torch.long)
        device = next(model.parameters()).device
        idx = idx.to(device)

        # Получаем скрытые состояния всех музыкантов через модель
        B, T = idx.shape
        tok = model.tok_emb(idx)
        pos = model.pos_emb[:, :T, :]
        shared_state = model.emb_drop(tok + pos)

        attn_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        # Прогоняем через все раунды репетиций
        hiddens_dict = {}
        for r, conductor in enumerate(model.conductors):
            contributions = []
            for name in model._musician_order:
                musician = model.musicians[name]
                contrib, hidden = musician(shared_state, attn_mask=attn_mask)
                contributions.append(contrib)
                # Запоминаем скрытые состояния последнего раунда
                if r == len(model.conductors) - 1:
                    hiddens_dict[name] = hidden

            orchestrated, _ = conductor(shared_state, contributions)
            gate = torch.sigmoid(model.rehearsal_gates[r])
            shared_state = shared_state + gate * orchestrated

        # Цепочка перевода: formalist → archetypist → algorithmist → linguist → formalist
        cycle = ['formalist', 'archetypist', 'algorithmist', 'linguist', 'formalist']
        results = {name: [] for name in MUSICIAN_NAMES}

        print("=" * 70)
        print(f"  КРОСС-ПЕРЕВОД: \"{text}\"")
        print("=" * 70)

        # Шаг 0: исходные выходы каждого музыканта (из модели напрямую)
        print("\n--- Исходные представления (от модели) ---")
        for name in MUSICIAN_NAMES:
            spec_logits = model.musicians[name].get_spec_logits(hiddens_dict[name])
            token_ids_out = spec_logits.argmax(dim=-1)[0].tolist()
            decoded = self._decode_tokens(name, token_ids_out)
            results[name].append(f"[исходное] {decoded}")
            print(f"  {self._DISPLAY_NAMES[name]}:")
            print(f"    {decoded}")

        # Цепочка переводов
        print("\n--- Цепочка переводов ---")
        h = hiddens_dict[cycle[0]]

        for step in range(len(cycle) - 1):
            src = cycle[step]
            tgt = cycle[step + 1]

            # Переводим
            logits = translator.translate(src, tgt, h)
            token_ids_out = logits.argmax(dim=-1)[0].tolist()
            decoded = self._decode_tokens(tgt, token_ids_out)

            results[tgt].append(f"[шаг {step + 1}: {src}→{tgt}] {decoded}")

            print(f"\n  Шаг {step + 1}: {self._DISPLAY_NAMES[src]} → {self._DISPLAY_NAMES[tgt]}")
            print(f"    Результат: {decoded}")

            # Подготавливаем hidden для следующего звена
            h = translator._logits_to_hidden(tgt, logits)

        print("\n" + "=" * 70)

        # Проверяем цикловую согласованность
        original_h = hiddens_dict[cycle[0]]
        l2_dist = F.mse_loss(h, original_h).item()
        print(f"  Цикловая ошибка (L2): {l2_dist:.6f}")
        print("=" * 70)

        return results

    def _decode_tokens(self, musician_name: str,
                       token_ids: List[int]) -> str:
        """Декодирует ID токенов в читаемую строку.

        Для формалиста, архетиписта и алгоритмиста использует
        специализированные словари. Для лингвиста — просто ID,
        так как его словарь задаётся внешним токенизатором.
        """
        if musician_name in self.vocabs:
            vocab = self.vocabs[musician_name]
            return vocab.decode_str(token_ids)
        else:
            # Лингвист — показываем ID (реальный декодинг зависит от токенизатора)
            # Убираем нули (pad)
            filtered = [str(t) for t in token_ids if t != 0]
            return '[' + ', '.join(filtered[:20]) + (', ...' if len(filtered) > 20 else '') + ']'

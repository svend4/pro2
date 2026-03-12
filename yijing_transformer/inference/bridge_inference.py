"""
Bridge: Продвинутые инференс-утилиты из utils_v12..v52.

Собирает все техники генерации и инференса,
дополняя базовый generate.py продвинутыми стратегиями.

Источники:
  v12: Dynamic Temperature — адаптивная температура по энтропии
  v46: Attention Sink Cache — КВ-кэш с attention sinks
  v46: Speculative Decoding Helper — спекулятивная генерация
  v48: Beam Search — поиск лучей
  v48: Nucleus Sampling — top-p/top-k сэмплирование
  v48: Repetition Penalty — штраф за повторы
  v48: Temperature Scheduler — динамическая температура при генерации
  v48: KV Cache Manager — управление KV-кэшем

Использование:
    from inference.bridge_inference import AdvancedGenerator
    gen = AdvancedGenerator(model, tokenizer)
    text = gen.generate("Once upon", strategy='beam', num_beams=4)
    text = gen.generate("Once upon", strategy='nucleus', top_p=0.9)
    text = gen.generate("Once upon", strategy='speculative', draft_model=small_model)
"""

from training.utils_v12 import dynamic_temperature
from training.utils_v46 import AttentionSinkCache, SpeculativeDecodingHelper
from training.utils_v48 import (
    BeamSearch,
    NucleusSampler,
    RepetitionPenalty,
    TemperatureScheduler,
    KVCacheManager,
)

import torch
import torch.nn.functional as F


class AdvancedGenerator:
    """Продвинутый генератор с поддержкой нескольких стратегий.

    Стратегии:
    - 'greedy': жадная генерация (baseline)
    - 'nucleus': top-p + top-k sampling (по умолчанию)
    - 'beam': beam search (качественнее, но медленнее)
    - 'speculative': спекулятивная генерация с draft model (быстрее)
    - 'dynamic_temp': адаптивная температура на основе энтропии
    """

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

        # Компоненты
        self.nucleus = NucleusSampler()
        self.rep_penalty = RepetitionPenalty()
        self.temp_scheduler = TemperatureScheduler()
        self.kv_manager = KVCacheManager()

    @torch.no_grad()
    def generate(self, prompt, max_tokens=100, strategy='nucleus', **kwargs):
        """Генерация текста с выбранной стратегией.

        Args:
            prompt: строка или list[int] — входной промпт
            max_tokens: максимальное число генерируемых токенов
            strategy: 'greedy' | 'nucleus' | 'beam' | 'speculative' | 'dynamic_temp'
            **kwargs: параметры для конкретной стратегии

        Returns:
            str: сгенерированный текст
        """
        self.model.eval()

        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = prompt

        if strategy == 'greedy':
            return self._generate_greedy(input_ids, max_tokens, **kwargs)
        elif strategy == 'nucleus':
            return self._generate_nucleus(input_ids, max_tokens, **kwargs)
        elif strategy == 'beam':
            return self._generate_beam(input_ids, max_tokens, **kwargs)
        elif strategy == 'speculative':
            return self._generate_speculative(input_ids, max_tokens, **kwargs)
        elif strategy == 'dynamic_temp':
            return self._generate_dynamic_temp(input_ids, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _generate_greedy(self, input_ids, max_tokens, **kwargs):
        """Жадная генерация — выбираем argmax."""
        context = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        block_size = self.model.cfg.block_size

        for _ in range(max_tokens):
            idx_cond = context[:, -block_size:]
            logits, _, _ = self.model(idx_cond)
            next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            context = torch.cat([context, next_token], dim=1)

        return self.tokenizer.decode(context[0].tolist()[len(input_ids):])

    def _generate_nucleus(self, input_ids, max_tokens,
                          temperature=0.7, top_k=50, top_p=0.9,
                          repetition_penalty=1.2, **kwargs):
        """Nucleus (top-p) sampling с repetition penalty."""
        context = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        block_size = self.model.cfg.block_size
        generated = []

        for _ in range(max_tokens):
            idx_cond = context[:, -block_size:]
            logits, _, _ = self.model(idx_cond)
            logits = logits[0, -1].clone()

            # Repetition penalty
            if repetition_penalty != 1.0:
                logits = self.rep_penalty.apply(
                    logits, context[0].tolist(),
                    penalty=repetition_penalty,
                )

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            # Top-K
            if top_k > 0:
                v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                probs[probs < v[-1]] = 0.0

            # Top-P
            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                probs[sorted_idx[mask]] = 0.0

            probs = probs / probs.sum()
            next_token = torch.multinomial(probs, 1).unsqueeze(0)
            context = torch.cat([context, next_token], dim=1)
            generated.append(next_token.item())

        return self.tokenizer.decode(generated)

    def _generate_beam(self, input_ids, max_tokens,
                       num_beams=4, length_penalty=1.0, **kwargs):
        """Beam search — поиск лучей для более качественной генерации."""
        beam_search = BeamSearch(
            model=self.model,
            beam_size=num_beams,
            max_length=max_tokens,
            length_penalty=length_penalty,
        )
        result_ids = beam_search.search(input_ids, self.device)
        return self.tokenizer.decode(result_ids)

    def _generate_speculative(self, input_ids, max_tokens,
                              draft_model=None, gamma=5, **kwargs):
        """Спекулятивная генерация с draft model.

        Draft model генерирует gamma токенов, target model верифицирует.
        Ускорение ~2-3x при хорошем draft model.
        """
        if draft_model is None:
            # Fallback на nucleus если нет draft model
            return self._generate_nucleus(input_ids, max_tokens, **kwargs)

        spec_helper = SpeculativeDecodingHelper(
            target_model=self.model,
            draft_model=draft_model,
            gamma=gamma,
        )
        result_ids = spec_helper.generate(input_ids, max_tokens, self.device)
        return self.tokenizer.decode(result_ids)

    def _generate_dynamic_temp(self, input_ids, max_tokens,
                               base_temp=0.7, **kwargs):
        """Генерация с динамической температурой на основе энтропии.

        Высокая энтропия → низкая температура (модель неуверена → аккуратнее).
        Низкая энтропия → высокая температура (модель уверена → можно рисковать).
        """
        context = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        block_size = self.model.cfg.block_size
        generated = []

        for _ in range(max_tokens):
            idx_cond = context[:, -block_size:]
            logits, _, _ = self.model(idx_cond)
            logits = logits[0, -1].clone()

            # Динамическая температура
            temp = dynamic_temperature(logits, base_temp=base_temp)
            logits = logits / temp
            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, 1).unsqueeze(0)
            context = torch.cat([context, next_token], dim=1)
            generated.append(next_token.item())

        return self.tokenizer.decode(generated)

    def generate_with_kv_cache(self, prompt, max_tokens=100, **kwargs):
        """Генерация с KV-кэшем (быстрее для длинных контекстов)."""
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = prompt

        context = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        kv_cache = None
        generated = []

        for _ in range(max_tokens):
            logits, _, kv_cache = self.model(
                context if kv_cache is None else context[:, -1:],
                kv_cache=kv_cache,
            )
            logits = logits[0, -1].clone()
            logits = logits / kwargs.get('temperature', 0.7)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).unsqueeze(0)
            context = torch.cat([context, next_token], dim=1)
            generated.append(next_token.item())

        return self.tokenizer.decode(generated)


def list_strategies():
    """Возвращает список доступных стратегий генерации."""
    return ['greedy', 'nucleus', 'beam', 'speculative', 'dynamic_temp']

"""
v48 утилиты: Beam Search, Top-k/Top-p Sampling,
Repetition Penalty, Temperature Scheduler, KV Cache Manager.

Beam Search: декодирование с beam search.
Ref: Standard NLP decoding (Graves, 2012)

Top-k / Top-p: nucleus sampling.
Ref: Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020)

Repetition Penalty: штраф за повторение токенов.
Ref: Keskar et al., "CTRL" (2019)

Temperature Scheduler: динамическая температура при генерации.
Ref: Common practice in LLM inference.

KV Cache Manager: управление key-value кэшем.
Ref: Standard transformer inference optimization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Beam Search ====================

class BeamSearch:
    """
    Beam Search декодирование.

    Поддерживает K лучших гипотез на каждом шаге.
    Length penalty для предотвращения bias к коротким seq.

    Args:
        beam_size: число лучей
        max_length: максимальная длина
        length_penalty: α для length normalization (score / len^α)
        eos_token_id: ID конца последовательности
    """
    def __init__(self, beam_size=4, max_length=128,
                 length_penalty=0.6, eos_token_id=2):
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.eos_token_id = eos_token_id

    def search(self, model, input_ids):
        """
        Beam search декодирование.

        Args:
            model: callable(input_ids) -> logits (..., vocab_size)
            input_ids: (1, T) начальные токены

        Returns:
            dict: {sequences, scores, best_sequence, best_score}
        """
        device = input_ids.device
        B = self.beam_size

        # Initialize beams: (beam_size, T)
        beams = input_ids.expand(B, -1).clone()
        scores = torch.zeros(B, device=device)
        finished = []

        model.eval()
        with torch.no_grad():
            for step in range(self.max_length):
                logits = model(beams)
                if isinstance(logits, tuple):
                    logits = logits[0]

                # Get last position logits: (B, V)
                if logits.dim() == 3:
                    last_logits = logits[:, -1, :]
                else:
                    last_logits = logits

                log_probs = F.log_softmax(last_logits, dim=-1)
                V = log_probs.size(-1)

                # Expand scores: (B, V)
                next_scores = scores.unsqueeze(-1) + log_probs

                # Flatten and get top-K
                flat_scores = next_scores.reshape(-1)
                topk_scores, topk_ids = flat_scores.topk(B)

                beam_ids = topk_ids // V
                token_ids = topk_ids % V

                # Update beams
                new_beams = torch.cat([
                    beams[beam_ids],
                    token_ids.unsqueeze(-1)
                ], dim=-1)
                scores = topk_scores

                # Check for EOS
                eos_mask = token_ids == self.eos_token_id
                for i in range(B):
                    if eos_mask[i]:
                        length = new_beams[i].size(0) - input_ids.size(1)
                        norm_score = scores[i].item() / (length ** self.length_penalty)
                        finished.append((new_beams[i].clone(), norm_score))

                # Filter out finished beams
                active = ~eos_mask
                if not active.any():
                    break
                beams = new_beams[active]
                scores = scores[active]

                # Pad if needed
                if beams.size(0) < B:
                    pad_n = B - beams.size(0)
                    beams = torch.cat([beams, beams[:pad_n]], dim=0)
                    scores = torch.cat([scores, torch.full((pad_n,), -1e9, device=device)])

        # Add remaining beams
        for i in range(beams.size(0)):
            length = beams[i].size(0) - input_ids.size(1)
            if length > 0:
                norm_score = scores[i].item() / (length ** self.length_penalty)
                finished.append((beams[i], norm_score))

        if not finished:
            return {
                'sequences': [input_ids[0]],
                'scores': [0.0],
                'best_sequence': input_ids[0],
                'best_score': 0.0,
            }

        # Sort by score
        finished.sort(key=lambda x: x[1], reverse=True)

        return {
            'sequences': [f[0] for f in finished],
            'scores': [f[1] for f in finished],
            'best_sequence': finished[0][0],
            'best_score': finished[0][1],
        }


# ==================== Top-k / Top-p Sampling ====================

class NucleusSampler:
    """
    Top-k и Top-p (nucleus) sampling.

    Top-k: оставляет K наиболее вероятных токенов.
    Top-p: оставляет минимальный набор с суммой p ≥ threshold.

    Args:
        top_k: число токенов для top-k (0 = отключено)
        top_p: порог для nucleus (1.0 = отключено)
        temperature: температура
        min_tokens_to_keep: минимум токенов
    """
    def __init__(self, top_k=50, top_p=0.9, temperature=1.0,
                 min_tokens_to_keep=1):
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.min_tokens_to_keep = min_tokens_to_keep

    def sample(self, logits):
        """
        Сэмплирует токен из логитов.

        Args:
            logits: (B, V) или (V,)

        Returns:
            dict: {token_ids, probs}
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # Temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Top-k filtering
        if self.top_k > 0:
            logits = self._top_k_filter(logits)

        # Top-p filtering
        if self.top_p < 1.0:
            logits = self._top_p_filter(logits)

        probs = F.softmax(logits, dim=-1)
        token_ids = torch.multinomial(probs, 1)

        return {
            'token_ids': token_ids.squeeze(-1),
            'probs': probs.gather(-1, token_ids).squeeze(-1),
        }

    def _top_k_filter(self, logits):
        """Keep only top-k tokens."""
        k = min(self.top_k, logits.size(-1))
        k = max(k, self.min_tokens_to_keep)
        top_k_values, _ = logits.topk(k, dim=-1)
        threshold = top_k_values[:, -1:]
        logits[logits < threshold] = -float('inf')
        return logits

    def _top_p_filter(self, logits):
        """Keep smallest set of tokens with cumsum >= top_p."""
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens above threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= self.top_p
        # Keep at least min_tokens_to_keep
        sorted_mask[:, :self.min_tokens_to_keep] = False

        # Scatter back
        mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits[mask] = -float('inf')
        return logits

    def get_info(self):
        return {
            'top_k': self.top_k,
            'top_p': self.top_p,
            'temperature': self.temperature,
        }


# ==================== Repetition Penalty ====================

class RepetitionPenalty:
    """
    Штраф за повторение токенов.

    Для уже сгенерированных токенов:
    - Если logit > 0: logit /= penalty
    - Если logit < 0: logit *= penalty

    Args:
        penalty: множитель штрафа (1.0 = нет, 1.2 типично)
        window_size: окно для проверки повторений (0 = вся seq)
    """
    def __init__(self, penalty=1.2, window_size=0):
        self.penalty = penalty
        self.window_size = window_size

    def apply(self, logits, generated_ids):
        """
        Применяет штраф к логитам.

        Args:
            logits: (B, V) логиты
            generated_ids: (B, T) уже сгенерированные токены

        Returns:
            Tensor: (B, V) модифицированные логиты
        """
        if self.penalty == 1.0:
            return logits

        logits = logits.clone()
        B = logits.size(0)

        for b in range(B):
            if self.window_size > 0:
                tokens = generated_ids[b, -self.window_size:]
            else:
                tokens = generated_ids[b]

            unique_tokens = tokens.unique()

            for token in unique_tokens:
                if token < 0:
                    continue
                if logits[b, token] > 0:
                    logits[b, token] /= self.penalty
                else:
                    logits[b, token] *= self.penalty

        return logits

    def apply_frequency_penalty(self, logits, generated_ids, freq_penalty=0.0):
        """
        Частотный штраф (линейный, по количеству появлений).

        Args:
            logits: (B, V)
            generated_ids: (B, T)
            freq_penalty: множитель частотного штрафа

        Returns:
            Tensor: (B, V)
        """
        if freq_penalty == 0.0:
            return logits

        logits = logits.clone()
        B, V = logits.shape

        for b in range(B):
            counts = torch.zeros(V, device=logits.device)
            for t in generated_ids[b]:
                if 0 <= t < V:
                    counts[t] += 1
            logits[b] -= freq_penalty * counts

        return logits


# ==================== Temperature Scheduler ====================

class TemperatureScheduler:
    """
    Динамическая температура при генерации.

    Позволяет менять температуру в зависимости от
    позиции / confidence / entropy.

    Args:
        base_temperature: начальная температура
        mode: 'constant', 'linear_decay', 'entropy_adaptive'
        min_temperature: минимальная температура
    """
    def __init__(self, base_temperature=1.0, mode='constant',
                 min_temperature=0.1):
        self.base_temperature = base_temperature
        self.mode = mode
        self.min_temperature = min_temperature
        self._step = 0

    def get_temperature(self, logits=None, max_steps=100):
        """
        Вычисляет температуру для текущего шага.

        Args:
            logits: (B, V) для entropy-adaptive mode
            max_steps: для linear_decay

        Returns:
            float: температура
        """
        self._step += 1

        if self.mode == 'constant':
            return self.base_temperature

        elif self.mode == 'linear_decay':
            progress = min(self._step / max(max_steps, 1), 1.0)
            t = self.base_temperature - progress * (self.base_temperature - self.min_temperature)
            return max(t, self.min_temperature)

        elif self.mode == 'entropy_adaptive' and logits is not None:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
            max_entropy = math.log(logits.size(-1))
            normalized_entropy = entropy / max(max_entropy, 1e-10)

            # High entropy → lower temp (more focused)
            # Low entropy → higher temp (more exploration)
            t = self.base_temperature * (1.0 - 0.5 * normalized_entropy)
            return max(t, self.min_temperature)

        return self.base_temperature

    def apply(self, logits, max_steps=100):
        """
        Применяет температуру к логитам.

        Args:
            logits: (B, V)

        Returns:
            dict: {logits, temperature}
        """
        temp = self.get_temperature(logits, max_steps)
        return {
            'logits': logits / temp,
            'temperature': temp,
        }

    def reset(self):
        self._step = 0


# ==================== KV Cache Manager ====================

class KVCacheManager:
    """
    Управление Key-Value кэшем для трансформера.

    Эффективное хранение и обновление KV пар
    для авторегрессивной генерации.

    Args:
        n_layers: число слоёв
        max_length: максимальная длина кэша
    """
    def __init__(self, n_layers=6, max_length=2048):
        self.n_layers = n_layers
        self.max_length = max_length
        self._cache = {}
        self._total_length = 0

    def get(self, layer_idx):
        """
        Получает KV кэш для слоя.

        Returns:
            tuple: (key, value) или None
        """
        if layer_idx in self._cache:
            return self._cache[layer_idx]['key'], self._cache[layer_idx]['value']
        return None

    def update(self, layer_idx, new_key, new_value):
        """
        Обновляет кэш новыми KV.

        Args:
            layer_idx: индекс слоя
            new_key: (B, H, T_new, D)
            new_value: (B, H, T_new, D)

        Returns:
            tuple: (full_key, full_value)
        """
        if layer_idx not in self._cache:
            self._cache[layer_idx] = {'key': new_key, 'value': new_value}
        else:
            old_k = self._cache[layer_idx]['key']
            old_v = self._cache[layer_idx]['value']
            self._cache[layer_idx]['key'] = torch.cat([old_k, new_key], dim=2)
            self._cache[layer_idx]['value'] = torch.cat([old_v, new_value], dim=2)

        # Trim if exceeds max length
        cache_len = self._cache[layer_idx]['key'].size(2)
        if cache_len > self.max_length:
            excess = cache_len - self.max_length
            self._cache[layer_idx]['key'] = self._cache[layer_idx]['key'][:, :, excess:, :]
            self._cache[layer_idx]['value'] = self._cache[layer_idx]['value'][:, :, excess:, :]

        self._total_length = self._cache[layer_idx]['key'].size(2)
        return self._cache[layer_idx]['key'], self._cache[layer_idx]['value']

    def clear(self):
        """Очистить весь кэш."""
        self._cache.clear()
        self._total_length = 0

    def clear_layer(self, layer_idx):
        """Очистить кэш для конкретного слоя."""
        if layer_idx in self._cache:
            del self._cache[layer_idx]

    def get_length(self):
        """Текущая длина кэша."""
        return self._total_length

    def get_memory_bytes(self):
        """Приблизительный объём памяти кэша в байтах."""
        total = 0
        for layer_data in self._cache.values():
            total += layer_data['key'].nelement() * layer_data['key'].element_size()
            total += layer_data['value'].nelement() * layer_data['value'].element_size()
        return total

    def get_info(self):
        return {
            'n_layers_cached': len(self._cache),
            'total_length': self._total_length,
            'max_length': self.max_length,
            'memory_mb': self.get_memory_bytes() / (1024 * 1024),
        }

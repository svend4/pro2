"""
v46 утилиты: Token-level Loss Weighting, Sequence Packing,
Dynamic Padding, Attention Sink Cache, Speculative Decoding Helper.

Token Loss Weighting: разный вес для разных токенов.
Ref: Common practice in LLM training pipelines.

Sequence Packing: упаковка коротких seq в одну длинную.
Ref: Krell et al., "Efficient Sequence Packing" (2021)

Dynamic Padding: минимальный padding для батча.
Ref: Standard efficient training technique.

Attention Sink: кэширование первых токенов.
Ref: Xiao et al., "Efficient Streaming LLMs with Attention Sinks" (2023)

Speculative Decoding: ускорение инференса через draft model.
Ref: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Token-level Loss Weighting ====================

class TokenLossWeighter:
    """
    Взвешивание loss по типу/позиции токенов.

    Позволяет давать больший вес:
    - Редким токенам
    - Токенам в конце последовательности
    - Ключевым токенам (по маске)

    Args:
        mode: 'uniform', 'position', 'frequency', 'custom'
        position_decay: параметр экспоненциального роста по позиции
    """
    def __init__(self, mode='uniform', position_decay=0.0):
        self.mode = mode
        self.position_decay = position_decay
        self._freq_counts = None

    def compute_weights(self, targets, mask=None, vocab_size=None):
        """
        Вычисляет веса для каждого токена.

        Args:
            targets: (B, T) target token ids
            mask: (B, T) optional custom weights
            vocab_size: размер словаря (для frequency mode)

        Returns:
            Tensor: (B, T) weights
        """
        B, T = targets.shape
        device = targets.device

        if self.mode == 'uniform':
            weights = torch.ones(B, T, device=device)

        elif self.mode == 'position':
            # Линейно растущий вес: последние токены важнее
            pos = torch.arange(T, device=device, dtype=torch.float)
            weights = 1.0 + self.position_decay * pos / T
            weights = weights.unsqueeze(0).expand(B, -1)

        elif self.mode == 'frequency':
            # Inverse frequency weighting
            weights = self._frequency_weights(targets, vocab_size or int(targets.max()) + 1)

        elif self.mode == 'custom' and mask is not None:
            weights = mask.float()

        else:
            weights = torch.ones(B, T, device=device)

        return weights

    def _frequency_weights(self, targets, vocab_size):
        """Inverse frequency weights."""
        B, T = targets.shape
        # Count frequencies in batch
        counts = torch.zeros(vocab_size, device=targets.device)
        counts.scatter_add_(0, targets.reshape(-1),
                           torch.ones(B * T, device=targets.device))
        # Inverse frequency (with smoothing)
        freq = counts / max(counts.sum().item(), 1)
        inv_freq = 1.0 / (freq + 1e-6)
        inv_freq = inv_freq / inv_freq.mean()  # normalize
        # Map to targets
        weights = inv_freq[targets]
        return weights

    def weighted_loss(self, logits, targets, weights=None):
        """
        Cross-entropy с весами.

        Args:
            logits: (B, T, C) или (B, C)
            targets: (B, T) или (B,)
            weights: (B, T) или None

        Returns:
            Tensor: weighted loss
        """
        if logits.dim() == 3:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.reshape(-1, C), targets.reshape(-1), reduction='none'
            ).reshape(B, T)
        else:
            loss = F.cross_entropy(logits, targets, reduction='none')

        if weights is None:
            weights = self.compute_weights(targets)

        return (loss * weights).sum() / weights.sum()


# ==================== Sequence Packing ====================

class SequencePacker:
    """
    Упаковка нескольких коротких последовательностей в одну.

    Экономит compute, избегая padding.
    Создаёт attention mask для предотвращения cross-attention.

    Args:
        max_length: максимальная длина упакованной последовательности
        pad_token_id: ID pad токена
    """
    def __init__(self, max_length=512, pad_token_id=0):
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def pack(self, sequences):
        """
        Упаковывает список последовательностей.

        Args:
            sequences: list[Tensor] — список 1D тензоров

        Returns:
            dict: {
                packed_ids: (N, max_length),
                attention_mask: (N, max_length),
                position_ids: (N, max_length),
                sequence_ids: (N, max_length),  # какой seq принадлежит токен
                n_packed: int
            }
        """
        packed_batches = []
        current_batch = []
        current_len = 0

        for seq in sequences:
            seq_len = len(seq)
            if seq_len > self.max_length:
                seq = seq[:self.max_length]
                seq_len = self.max_length

            if current_len + seq_len > self.max_length and current_batch:
                packed_batches.append(current_batch)
                current_batch = []
                current_len = 0

            current_batch.append(seq)
            current_len += seq_len

        if current_batch:
            packed_batches.append(current_batch)

        # Build tensors
        all_ids = []
        all_masks = []
        all_pos = []
        all_seq_ids = []

        for batch in packed_batches:
            ids = []
            positions = []
            seq_ids = []

            for seq_idx, seq in enumerate(batch):
                ids.extend(seq.tolist() if isinstance(seq, torch.Tensor) else seq)
                positions.extend(range(len(seq)))
                seq_ids.extend([seq_idx] * len(seq))

            # Pad
            pad_len = self.max_length - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [self.pad_token_id] * pad_len
            positions = positions + [0] * pad_len
            seq_ids = seq_ids + [-1] * pad_len

            all_ids.append(ids)
            all_masks.append(mask)
            all_pos.append(positions)
            all_seq_ids.append(seq_ids)

        device = sequences[0].device if isinstance(sequences[0], torch.Tensor) else 'cpu'
        return {
            'packed_ids': torch.tensor(all_ids, device=device),
            'attention_mask': torch.tensor(all_masks, device=device),
            'position_ids': torch.tensor(all_pos, device=device),
            'sequence_ids': torch.tensor(all_seq_ids, device=device),
            'n_packed': len(packed_batches),
        }

    def create_block_attention_mask(self, sequence_ids):
        """
        Создаёт block-diagonal attention mask.

        Токены могут attend только к токенам той же последовательности.

        Args:
            sequence_ids: (B, T) — id последовательности для каждого токена

        Returns:
            Tensor: (B, T, T) attention mask
        """
        B, T = sequence_ids.shape
        # (B, T, 1) == (B, 1, T) → (B, T, T)
        mask = sequence_ids.unsqueeze(-1) == sequence_ids.unsqueeze(-2)
        # Exclude padding (-1 == -1 should be False)
        valid = (sequence_ids >= 0).unsqueeze(-1) & (sequence_ids >= 0).unsqueeze(-2)
        return (mask & valid).float()


# ==================== Dynamic Padding ====================

class DynamicPadder:
    """
    Минимальный padding для каждого батча.

    Вместо padding до max_length, pad до максимальной
    длины в текущем батче.

    Args:
        pad_token_id: ID pad токена
        pad_to_multiple: выравнивание (8 для tensor cores)
    """
    def __init__(self, pad_token_id=0, pad_to_multiple=8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple = pad_to_multiple
        self._stats = {'total_tokens': 0, 'padded_tokens': 0}

    def pad_batch(self, sequences):
        """
        Dynamic padding для батча.

        Args:
            sequences: list[Tensor] — список 1D тензоров разной длины

        Returns:
            dict: {input_ids, attention_mask, lengths}
        """
        lengths = [len(s) for s in sequences]
        max_len = max(lengths)

        # Round up to multiple
        if self.pad_to_multiple > 1:
            max_len = ((max_len + self.pad_to_multiple - 1)
                       // self.pad_to_multiple * self.pad_to_multiple)

        B = len(sequences)
        device = sequences[0].device if isinstance(sequences[0], torch.Tensor) else 'cpu'

        input_ids = torch.full((B, max_len), self.pad_token_id,
                               dtype=torch.long, device=device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)

        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            if isinstance(seq, torch.Tensor):
                input_ids[i, :seq_len] = seq
            else:
                input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :seq_len] = 1

        self._stats['total_tokens'] += B * max_len
        self._stats['padded_tokens'] += B * max_len - sum(lengths)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'lengths': lengths,
        }

    def get_efficiency(self):
        """Процент полезных (не-padding) токенов."""
        total = self._stats['total_tokens']
        if total == 0:
            return 1.0
        return 1.0 - self._stats['padded_tokens'] / total


# ==================== Attention Sink Cache ====================

class AttentionSinkCache:
    """
    Кэш attention sinks для efficient streaming.

    Сохраняет первые N токенов (sinks) + последние M (window).
    Позволяет streaming inference без потери качества.

    Args:
        n_sink_tokens: число sink токенов (обычно 4)
        window_size: размер sliding window
    """
    def __init__(self, n_sink_tokens=4, window_size=256):
        self.n_sink_tokens = n_sink_tokens
        self.window_size = window_size
        self._cache = {}

    def update(self, layer_idx, key, value):
        """
        Обновляет кэш для слоя.

        Args:
            layer_idx: индекс слоя
            key: (B, H, T, D) ключи
            value: (B, H, T, D) значения

        Returns:
            dict: {key, value} — обновлённый кэш
        """
        T = key.size(2)

        if layer_idx not in self._cache:
            # First call — save everything
            self._cache[layer_idx] = {'key': key, 'value': value}
        else:
            # Append new tokens
            old_k = self._cache[layer_idx]['key']
            old_v = self._cache[layer_idx]['value']
            key = torch.cat([old_k, key], dim=2)
            value = torch.cat([old_v, value], dim=2)

        total_len = key.size(2)
        max_cache = self.n_sink_tokens + self.window_size

        if total_len > max_cache:
            # Keep sinks + recent window
            sink_k = key[:, :, :self.n_sink_tokens, :]
            sink_v = value[:, :, :self.n_sink_tokens, :]
            window_k = key[:, :, -self.window_size:, :]
            window_v = value[:, :, -self.window_size:, :]
            key = torch.cat([sink_k, window_k], dim=2)
            value = torch.cat([sink_v, window_v], dim=2)

        self._cache[layer_idx] = {'key': key, 'value': value}
        return {'key': key, 'value': value}

    def get(self, layer_idx):
        """Получить кэш для слоя."""
        return self._cache.get(layer_idx, None)

    def clear(self):
        """Очистить кэш."""
        self._cache.clear()

    def get_info(self):
        info = {
            'n_layers_cached': len(self._cache),
            'n_sink_tokens': self.n_sink_tokens,
            'window_size': self.window_size,
        }
        if self._cache:
            first = next(iter(self._cache.values()))
            info['cache_length'] = first['key'].size(2)
        return info


# ==================== Speculative Decoding Helper ====================

class SpeculativeDecodingHelper:
    """
    Вспомогательные функции для speculative decoding.

    Draft model генерирует K токенов, target model
    верифицирует их за один forward pass.

    Args:
        n_speculative: число спекулятивных токенов
        temperature: температура сэмплирования
    """
    def __init__(self, n_speculative=4, temperature=1.0):
        self.n_speculative = n_speculative
        self.temperature = temperature
        self._stats = {'total_drafted': 0, 'total_accepted': 0}

    def draft_tokens(self, draft_model, input_ids, n_tokens=None):
        """
        Генерирует спекулятивные токены через draft model.

        Args:
            draft_model: маленькая модель
            input_ids: (B, T) входные токены
            n_tokens: число токенов (default: n_speculative)

        Returns:
            dict: {tokens, logits}
        """
        n = n_tokens or self.n_speculative
        tokens = []
        logits_list = []
        current = input_ids

        draft_model.eval()
        with torch.no_grad():
            for _ in range(n):
                output = draft_model(current)
                if isinstance(output, tuple):
                    logit = output[0]
                else:
                    logit = output

                # Last position logits
                if logit.dim() == 3:
                    last_logit = logit[:, -1, :]
                else:
                    last_logit = logit

                # Sample
                if self.temperature > 0:
                    probs = F.softmax(last_logit / self.temperature, dim=-1)
                    token = torch.multinomial(probs, 1)
                else:
                    token = last_logit.argmax(dim=-1, keepdim=True)

                tokens.append(token)
                logits_list.append(last_logit)
                current = torch.cat([current, token], dim=-1)

        return {
            'tokens': torch.cat(tokens, dim=-1),
            'logits': torch.stack(logits_list, dim=1),
        }

    def verify_tokens(self, target_logits, draft_logits, draft_tokens):
        """
        Верификация спекулятивных токенов.

        Принимает токен если p_target(x) / p_draft(x) >= random.

        Args:
            target_logits: (B, K, V) логиты target model
            draft_logits: (B, K, V) логиты draft model
            draft_tokens: (B, K) предложенные токены

        Returns:
            dict: {accepted_tokens, n_accepted, acceptance_rate}
        """
        B, K, V = target_logits.shape

        target_probs = F.softmax(target_logits / self.temperature, dim=-1)
        draft_probs = F.softmax(draft_logits / self.temperature, dim=-1)

        accepted = []
        n_accepted = 0

        for b in range(B):
            batch_accepted = []
            for k in range(K):
                token = draft_tokens[b, k].item()
                p_target = target_probs[b, k, token].item()
                p_draft = max(draft_probs[b, k, token].item(), 1e-10)
                ratio = p_target / p_draft
                r = torch.rand(1).item()

                if r < ratio:
                    batch_accepted.append(token)
                    n_accepted += 1
                else:
                    # Resample from adjusted distribution
                    adjusted = torch.clamp(target_probs[b, k] - draft_probs[b, k], min=0)
                    if adjusted.sum() > 0:
                        adjusted = adjusted / adjusted.sum()
                        new_token = torch.multinomial(adjusted, 1).item()
                    else:
                        new_token = target_probs[b, k].argmax().item()
                    batch_accepted.append(new_token)
                    break  # Stop accepting after first rejection

            accepted.append(batch_accepted)

        total_drafted = B * K
        self._stats['total_drafted'] += total_drafted
        self._stats['total_accepted'] += n_accepted

        return {
            'accepted_tokens': accepted,
            'n_accepted': n_accepted,
            'acceptance_rate': n_accepted / max(total_drafted, 1),
        }

    def get_stats(self):
        total = max(self._stats['total_drafted'], 1)
        return {
            'total_drafted': self._stats['total_drafted'],
            'total_accepted': self._stats['total_accepted'],
            'overall_acceptance_rate': self._stats['total_accepted'] / total,
        }

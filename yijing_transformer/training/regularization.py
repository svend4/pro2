"""
Token Merging (ToMe), Cosine Annealing with Warm Restarts, Gradient Noise.

Token Merging сокращает число токенов путём слияния похожих,
ускоряя FFN без потери качества (Bolya et al., 2023).

Cosine Annealing with Warm Restarts — scheduler с периодическими
перезапусками learning rate (Loshchilov & Hutter, 2017).

Gradient Noise добавляет гауссов шум к градиентам для лучшей
генерализации (Neelakantan et al., 2015).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Token Merging ====================

class TokenMerger(nn.Module):
    """
    Token Merging (ToMe): слияние похожих токенов для ускорения.

    Алгоритм:
    1. Разбиваем токены на две группы (чётные/нечётные)
    2. Вычисляем cosine similarity между группами
    3. Сливаем r наиболее похожих пар (среднее)
    4. После FFN — unmerge обратно

    Это сокращает вычисления в FFN пропорционально merge_ratio.
    """
    def __init__(self, merge_ratio=0.25):
        super().__init__()
        self.merge_ratio = merge_ratio

    def forward(self, x):
        """
        Merge токенов.

        Args:
            x: (B, T, D)

        Returns:
            merged: (B, T - r, D) — сжатая последовательность
            unmerge_info: данные для восстановления
        """
        B, T, D = x.shape
        r = int(T * self.merge_ratio)
        if r == 0 or T <= 2:
            return x, None

        # Разбиваем на A (чётные) и B (нечётные)
        a_idx = torch.arange(0, T, 2, device=x.device)[:T // 2]
        b_idx = torch.arange(1, T, 2, device=x.device)[:T // 2]

        a = x[:, a_idx]  # (B, T//2, D)
        b = x[:, b_idx]  # (B, T//2, D)

        # Cosine similarity
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)

        # Для каждого a[i], найти наиболее похожий b[j]
        # Используем матрицу сходства
        scores = torch.bmm(a_norm, b_norm.transpose(1, 2))  # (B, T//2, T//2)

        # Жадный matching: для каждого a[i] берём лучший b[j]
        max_scores, max_idx = scores.max(dim=-1)  # (B, T//2)

        # Берём top-r пар с наибольшим сходством
        r = min(r, a.shape[1])
        _, merge_idx = max_scores.topk(r, dim=-1)  # (B, r)

        # Bipartite matching: для каждого a, берём лучший b (1-to-1)
        # Используем жадное 1-to-1 matching для consistent размера
        results = []
        unmerge_infos = []
        n_a = a.shape[1]
        n_b = b.shape[1]

        for batch in range(B):
            # Жадное 1-to-1 matching
            pairs = []
            used_b = set()
            # Сортируем a индексы по max_score (убывание)
            sorted_a = max_scores[batch].argsort(descending=True)
            for ai in sorted_a.tolist():
                bi = max_idx[batch, ai].item()
                if bi not in used_b and len(pairs) < r:
                    pairs.append((ai, bi))
                    used_b.add(bi)

            a_merge = set(p[0] for p in pairs)
            b_merge = set(p[1] for p in pairs)
            actual_r = len(pairs)

            merged = []
            merge_map = {}

            pos = 0
            a_keep = [i for i in range(n_a) if i not in a_merge]
            for i in a_keep:
                merged.append(a[batch, i])
                pos += 1

            merge_start = pos
            for ai, bi in pairs:
                avg = (a[batch, ai] + b[batch, bi]) / 2.0
                merged.append(avg)
                merge_map[pos] = (ai, bi)
                pos += 1

            b_keep = [i for i in range(n_b) if i not in b_merge]
            for i in b_keep:
                merged.append(b[batch, i])
                pos += 1

            results.append(torch.stack(merged))
            unmerge_infos.append({
                'a_keep': a_keep,
                'b_keep': b_keep,
                'merge_map': merge_map,
                'merge_start': merge_start,
                'a_idx': a_idx,
                'b_idx': b_idx,
                'orig_T': T,
                'r': actual_r,
            })

        # Pad to same length if needed (batch consistency)
        max_len = max(res.shape[0] for res in results)
        padded = []
        for r_tensor in results:
            if r_tensor.shape[0] < max_len:
                pad = torch.zeros(max_len - r_tensor.shape[0], D, device=x.device, dtype=x.dtype)
                padded.append(torch.cat([r_tensor, pad]))
            else:
                padded.append(r_tensor)

        merged_out = torch.stack(padded)
        return merged_out, unmerge_infos

    def unmerge(self, x, unmerge_info):
        """
        Восстановление исходной длины после FFN.

        Args:
            x: (B, T-r, D) — обработанная сжатая последовательность
            unmerge_info: данные из forward

        Returns:
            (B, T, D) — восстановленная последовательность
        """
        if unmerge_info is None:
            return x

        B, _, D = x.shape
        results = []

        for batch in range(B):
            info = unmerge_info[batch]
            T = info['orig_T']
            out = torch.zeros(T, D, device=x.device, dtype=x.dtype)
            a_idx = info['a_idx']
            b_idx = info['b_idx']

            pos = 0
            # a_keep
            for i in info['a_keep']:
                out[a_idx[i]] = x[batch, pos]
                pos += 1

            # merged pairs — дублируем в обе позиции
            for merge_pos, (ai, bi) in info['merge_map'].items():
                out[a_idx[ai]] = x[batch, pos]
                out[b_idx[bi]] = x[batch, pos]
                pos += 1

            # b_keep
            for i in info['b_keep']:
                out[b_idx[i]] = x[batch, pos]
                pos += 1

            results.append(out)

        return torch.stack(results)


# ==================== Cosine Annealing с Warm Restarts ====================

class CosineAnnealingWarmRestarts:
    """
    Cosine annealing с warm restarts.

    lr(t) = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * T_cur / T_i))

    После каждого цикла T_i может расти (T_mult > 1) для всё более
    длинных фаз обучения.

    Args:
        optimizer: PyTorch optimizer
        T_0: длина первого цикла (в шагах)
        T_mult: множитель длины цикла (1 = одинаковые, 2 = удваиваем)
        eta_min: минимальный learning rate
        warmup_steps: шаги warmup в начале каждого цикла
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=1e-6, warmup_steps=0):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0
        self.cycle = 0
        self.T_cur = 0
        self.T_i = T_0

    def step(self):
        """Обновляет learning rate."""
        self.step_count += 1
        self.T_cur += 1

        # Проверяем, нужен ли restart
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.cycle += 1
            self.T_i = int(self.T_0 * (self.T_mult ** self.cycle))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.T_cur < self.warmup_steps:
                # Linear warmup
                lr = base_lr * (self.T_cur + 1) / self.warmup_steps
            else:
                # Cosine decay
                progress = (self.T_cur - self.warmup_steps) / max(1, self.T_i - self.warmup_steps)
                lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
            pg['lr'] = lr

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'step_count': self.step_count, 'cycle': self.cycle,
            'T_cur': self.T_cur, 'T_i': self.T_i,
        }

    def load_state_dict(self, state):
        for k, v in state.items():
            setattr(self, k, v)


# ==================== Gradient Noise ====================

class GradientNoise:
    """
    Добавляет гауссов шум к градиентам: g' = g + N(0, sigma²)

    sigma² = eta / (1 + t)^gamma

    Шум уменьшается со временем. Помогает выходить из острых минимумов
    и находить более плоские (лучше генерализующие) решения.

    Ref: Neelakantan et al., "Adding Gradient Noise Improves Learning
    for Very Deep Networks" (2015)

    Args:
        eta: начальная амплитуда шума
        gamma: скорость убывания (0.55 по умолчанию)
    """
    def __init__(self, eta=0.01, gamma=0.55):
        self.eta = eta
        self.gamma = gamma
        self.step_count = 0

    def add_noise(self, model):
        """Добавляет шум ко всем градиентам модели."""
        self.step_count += 1
        sigma = math.sqrt(self.eta / (1 + self.step_count) ** self.gamma)

        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * sigma
                    param.grad.add_(noise)

        return sigma

    def state_dict(self):
        return {'step_count': self.step_count, 'eta': self.eta, 'gamma': self.gamma}

    def load_state_dict(self, state):
        for k, v in state.items():
            setattr(self, k, v)


# ==================== v51: Антиподальная регуляризация (Ступень 2) ====================

class AntipodalRegularization(nn.Module):
    """Антиподальная регуляризация для квантизатора (Фомюк + Герман).

    Штраф: ||Q(x) + Q(-x)|| → 0.
    Заставляет квантизатор «знать» об антиподальной структуре:
    hex(i) + hex(63-i) = 0 (покоординатно).

    Эквивалент weight tying: 64 кодовых слова работают как 128,
    потому что каждое слово несёт информацию о своём антиподе.

    Герман доказывает: n + n̄ = P+1 для упаковки с периодом P.
    """
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight

    def forward(self, quantizer, x):
        """Вычисляет антиподальный loss.

        Args:
            quantizer: квантизатор с методом forward(x)
            x: входной тензор (..., 6)

        Returns:
            scalar loss
        """
        q_x = quantizer(x)
        q_neg_x = quantizer(-x)
        # Антиподальность: Q(x) + Q(-x) ≈ 0
        antipodal_loss = (q_x + q_neg_x).pow(2).mean()
        return self.weight * antipodal_loss


class HexagramAntipodalLoss(nn.Module):
    """Регуляризация Фомюка: антиподальные пары гексаграмм балансируют кодбук.

    Применяется к эмбеддингам кодбука (не к слою attention — это ключевая
    правка по сравнению с реализацией v59).

    В Q6-гиперкубе гексаграмма i и гексаграмма (63-i) являются антиподами
    (все 6 бит инвертированы). Их эмбеддинги должны «уравновешивать» друг
    друга: emb(i) + emb(63-i) ≈ 0.

    Это гарантирует симметрию кодбука и предотвращает «схлопывание»
    всех гексаграмм в одну область пространства.

    Args:
        n_hexagrams: число гексаграмм (64 по умолчанию)
        weight: вес в суммарном loss (рекомендуется 0.001)
    """

    def __init__(self, n_hexagrams: int = 64, weight: float = 0.001):
        super().__init__()
        self.n_hexagrams = n_hexagrams
        self.weight = weight
        # Антиподальные пары: (0,63), (1,62), ..., (31,32)
        self.antipodal_pairs = [(i, n_hexagrams - 1 - i) for i in range(n_hexagrams // 2)]

    def forward(self, codebook: torch.Tensor) -> torch.Tensor:
        """Вычисляет антиподальный loss для кодбука.

        Args:
            codebook: (n_hexagrams, d_model) — эмбеддинги всех гексаграмм

        Returns:
            loss: скаляр — среднее ||emb(i) + emb(63-i)||² по всем парам
        """
        losses = []
        for i, j in self.antipodal_pairs:
            emb_i = codebook[i]
            emb_j = codebook[j]
            # Антиподы должны уравновешивать друг друга: сумма ≈ 0
            balance_loss = (emb_i + emb_j).pow(2).mean()
            losses.append(balance_loss)
        return self.weight * torch.stack(losses).mean()

    def diversity_score(self, codebook: torch.Tensor) -> float:
        """Метрика разнообразия кодбука (чем выше — тем лучше).

        Высокое разнообразие = антиподальные пары действительно противоположны.

        Returns:
            diversity: среднее косинусное сходство антиподальных пар
                       (близко к -1 = хорошо, близко к +1 = плохо)
        """
        scores = []
        for i, j in self.antipodal_pairs:
            e_i = torch.nn.functional.normalize(codebook[i], dim=0)
            e_j = torch.nn.functional.normalize(codebook[j], dim=0)
            scores.append((e_i * e_j).sum().item())
        return sum(scores) / len(scores) if scores else 0.0

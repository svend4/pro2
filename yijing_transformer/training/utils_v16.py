"""
v16 утилиты: Structured Pruning, Contrastive Loss, Sequence Packing,
PCGrad, Model Merging.

Structured Pruning: удаление целых каналов/нейронов (не отдельных весов).
В отличие от unstructured pruning, реально ускоряет inference.

Contrastive Loss: SimCLR-style loss для обучения представлений.
Полезно для pre-training и fine-tuning.

Sequence Packing: упаковка коротких последовательностей в один batch
без padding. Экономит compute на коротких примерах.

PCGrad: проецирует конфликтующие градиенты при multi-task обучении.
Ref: Yu et al., "Gradient Surgery for Multi-Task Learning" (2020)

Model Merging: объединение весов нескольких моделей.
Поддерживает: average, weighted average, SLERP, TIES.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Structured Pruning ====================

class StructuredPruner:
    """
    Structured pruning: удаляет целые нейроны/каналы по L1-норме.

    В отличие от head pruning (v14), работает на уровне FFN нейронов.
    Удаляет нейроны с наименьшей L1-нормой выходных весов.

    Args:
        model: YiJingGPT
        prune_ratio: доля нейронов для удаления (0-1)
    """
    def __init__(self, model, prune_ratio=0.2):
        self.model = model
        self.prune_ratio = prune_ratio

    def compute_importance(self):
        """
        Вычисляет importance score для каждого FFN нейрона.

        Returns:
            list[dict]: importance по слоям
        """
        results = []
        for i, layer in enumerate(self.model.core.layers):
            ffn = layer.ffn
            # Находим первый Linear в FFN (w1 или Sequential[0])
            if hasattr(ffn, 'w1'):
                weight = ffn.w1.weight  # (FFN_H, D)
            elif isinstance(ffn, nn.Sequential):
                weight = ffn[0].weight  # (FFN_H, D)
            else:
                results.append({'layer': i, 'importance': None})
                continue

            # L1-норма каждого нейрона (строки weight)
            importance = weight.abs().sum(dim=1)  # (FFN_H,)
            results.append({
                'layer': i,
                'importance': importance.detach(),
                'n_neurons': weight.shape[0],
            })

        return results

    def get_prune_mask(self):
        """
        Возвращает маску: True = сохранить, False = удалить.

        Returns:
            list[Tensor]: маски по слоям
        """
        importances = self.compute_importance()
        masks = []

        for info in importances:
            if info['importance'] is None:
                masks.append(None)
                continue

            imp = info['importance']
            n = len(imp)
            n_prune = int(n * self.prune_ratio)
            threshold = imp.topk(n - n_prune, largest=True).values[-1]
            mask = imp >= threshold
            masks.append(mask)

        return masks

    def apply_masks(self):
        """
        Применяет soft pruning: зануляет удалённые нейроны.

        Returns:
            dict: статистика прунинга
        """
        masks = self.get_prune_mask()
        total_pruned = 0
        total_neurons = 0

        with torch.no_grad():
            for i, (mask, layer) in enumerate(
                zip(masks, self.model.core.layers)
            ):
                if mask is None:
                    continue

                ffn = layer.ffn
                if hasattr(ffn, 'w1'):
                    ffn.w1.weight[~mask] = 0
                    if hasattr(ffn, 'w2'):
                        ffn.w2.weight[:, ~mask] = 0
                elif isinstance(ffn, nn.Sequential):
                    ffn[0].weight[~mask] = 0
                    # Последний Linear: зануляем входные каналы
                    for m in reversed(list(ffn.modules())):
                        if isinstance(m, nn.Linear):
                            m.weight[:, ~mask] = 0
                            break

                total_pruned += (~mask).sum().item()
                total_neurons += len(mask)

        return {
            'total_pruned': total_pruned,
            'total_neurons': total_neurons,
            'prune_ratio_actual': total_pruned / max(total_neurons, 1),
        }


# ==================== Contrastive Loss ====================

def contrastive_loss(z1, z2, temperature=0.5):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    SimCLR-style: для batch из N пар (z1[i], z2[i]),
    позитивная пара = (z1[i], z2[i]), негативные = все остальные.

    Args:
        z1: (N, D) — представления первого augmentation
        z2: (N, D) — представления второго augmentation
        temperature: температура

    Returns:
        loss: скаляр
    """
    N = z1.shape[0]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # Все пары сходства
    representations = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim_matrix = representations @ representations.T  # (2N, 2N)
    sim_matrix = sim_matrix / temperature

    # Маска: исключаем диагональ (self-similarity)
    mask = ~torch.eye(2 * N, dtype=torch.bool, device=z1.device)

    # Позитивные пары: (i, i+N) и (i+N, i)
    pos_sim = torch.cat([
        torch.diag(sim_matrix, N),   # z1[i] · z2[i]
        torch.diag(sim_matrix, -N),  # z2[i] · z1[i]
    ])  # (2N,)

    # Негативные: все кроме себя
    neg_sim = sim_matrix[mask].reshape(2 * N, -1)  # (2N, 2N-1)

    # Loss: -log(exp(pos) / sum(exp(neg)))
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (2N, 2N)
    labels = torch.zeros(2 * N, dtype=torch.long, device=z1.device)  # pos = index 0

    return F.cross_entropy(logits, labels)


# ==================== Sequence Packing ====================

class SequencePacker:
    """
    Упаковка коротких последовательностей в один batch без padding.

    Вместо padding до max_len, склеиваем последовательности
    друг за другом с разделителями. Создаём attention mask,
    запрещающий attention между разными документами.

    Args:
        max_seq_len: максимальная длина упакованной последовательности
        pad_id: ID padding токена
        sep_id: ID разделителя (None = не добавлять)
    """
    def __init__(self, max_seq_len=512, pad_id=0, sep_id=None):
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.sep_id = sep_id

    def pack(self, sequences):
        """
        Упаковывает список последовательностей в packed batches.

        Args:
            sequences: list[list[int]] — список токен-последовательностей

        Returns:
            list[dict]: packed batches, каждый содержит:
                - 'input_ids': (T,) — упакованные токены
                - 'doc_ids': (T,) — какому документу принадлежит каждый токен
                - 'n_docs': число документов в batch
        """
        batches = []
        current_ids = []
        current_doc_ids = []
        current_doc = 0

        for seq in sequences:
            needed = len(seq)
            if self.sep_id is not None:
                needed += 1

            # Не помещается → начинаем новый batch
            if len(current_ids) + needed > self.max_seq_len:
                if current_ids:
                    batches.append(self._finalize(current_ids, current_doc_ids, current_doc))
                current_ids = []
                current_doc_ids = []
                current_doc = 0

            # Разделитель
            if self.sep_id is not None and current_ids:
                current_ids.append(self.sep_id)
                current_doc_ids.append(-1)  # separator

            # Добавляем последовательность
            current_ids.extend(seq)
            current_doc_ids.extend([current_doc] * len(seq))
            current_doc += 1

        # Финальный batch
        if current_ids:
            batches.append(self._finalize(current_ids, current_doc_ids, current_doc))

        return batches

    def _finalize(self, ids, doc_ids, n_docs):
        """Финализирует один packed batch."""
        # Pad до max_seq_len
        pad_len = self.max_seq_len - len(ids)
        ids = ids + [self.pad_id] * pad_len
        doc_ids = doc_ids + [-1] * pad_len

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'doc_ids': torch.tensor(doc_ids, dtype=torch.long),
            'n_docs': n_docs,
        }

    @staticmethod
    def create_packing_mask(doc_ids):
        """
        Создаёт attention mask для packed sequence.

        Токены могут видеть только токены того же документа.

        Args:
            doc_ids: (T,) — ID документа для каждого токена

        Returns:
            mask: (T, T) — True = можно attend, False = нельзя
        """
        T = doc_ids.shape[0]
        # Токены одного документа могут видеть друг друга
        mask = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)  # (T, T)
        # Исключаем padding (-1 != -1 should be False for pad tokens)
        # Но -1 == -1 is True, поэтому маскируем pad отдельно
        valid = (doc_ids >= 0).unsqueeze(0) & (doc_ids >= 0).unsqueeze(1)
        mask = mask & valid
        # Causal
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=doc_ids.device))
        return mask & causal


# ==================== PCGrad ====================

class PCGrad:
    """
    Projected Conflicting Gradients (PCGrad).

    При multi-task обучении градиенты разных задач могут конфликтовать.
    PCGrad проецирует конфликтующие градиенты на нормальную плоскость,
    устраняя деструктивную интерференцию.

    Ref: Yu et al., "Gradient Surgery for Multi-Task Learning" (2020)

    Args:
        optimizer: PyTorch optimizer
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, losses):
        """
        PCGrad step: вычисляет градиенты для каждого loss,
        проецирует конфликтующие, суммирует.

        Args:
            losses: list[Tensor] — loss для каждой задачи
        """
        # Собираем градиенты для каждого loss
        task_grads = []
        for loss in losses:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grads = []
            for p in self._get_params():
                grads.append(p.grad.clone() if p.grad is not None
                           else torch.zeros_like(p))
            task_grads.append(grads)

        # Проецируем конфликтующие градиенты
        n_tasks = len(task_grads)
        projected = [list(g) for g in task_grads]  # deep copy

        for i in range(n_tasks):
            for j in range(n_tasks):
                if i == j:
                    continue
                # Проверяем конфликт для каждого параметра
                for k in range(len(projected[i])):
                    dot = (projected[i][k] * task_grads[j][k]).sum()
                    if dot < 0:
                        # Конфликт: проецируем
                        norm_sq = (task_grads[j][k] ** 2).sum().clamp(min=1e-12)
                        projected[i][k] = projected[i][k] - (dot / norm_sq) * task_grads[j][k]

        # Суммируем проецированные градиенты
        self.optimizer.zero_grad()
        for k, p in enumerate(self._get_params()):
            total_grad = sum(projected[i][k] for i in range(n_tasks))
            p.grad = total_grad

        self.optimizer.step()

    def _get_params(self):
        """Возвращает все параметры с requires_grad."""
        params = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    params.append(p)
        return params


# ==================== Model Merging ====================

def merge_models_average(models, weights=None):
    """
    Объединяет модели усреднением весов.

    Args:
        models: list[nn.Module] — модели для объединения
        weights: list[float] — веса (None = равные)

    Returns:
        merged state_dict
    """
    n = len(models)
    if weights is None:
        weights = [1.0 / n] * n

    assert len(weights) == n
    assert abs(sum(weights) - 1.0) < 1e-6

    merged = {}
    ref_sd = models[0].state_dict()

    for key in ref_sd:
        merged[key] = sum(
            w * m.state_dict()[key].float()
            for w, m in zip(weights, models)
        ).to(ref_sd[key].dtype)

    return merged


def merge_models_slerp(model_a, model_b, t=0.5):
    """
    Spherical Linear Interpolation (SLERP) между двумя моделями.

    SLERP сохраняет норму вектора (в отличие от linear interp),
    что лучше для объединения fine-tuned моделей.

    Args:
        model_a: первая модель
        model_b: вторая модель
        t: интерполяция (0 = model_a, 1 = model_b)

    Returns:
        merged state_dict
    """
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    merged = {}

    for key in sd_a:
        a = sd_a[key].float().flatten()
        b = sd_b[key].float().flatten()

        # Cosine similarity
        dot = (a * b).sum()
        norm_a = a.norm()
        norm_b = b.norm()

        if norm_a < 1e-8 or norm_b < 1e-8:
            # Degenerate: linear interpolation
            merged[key] = ((1 - t) * sd_a[key].float() + t * sd_b[key].float()
                          ).to(sd_a[key].dtype)
            continue

        cos_sim = (dot / (norm_a * norm_b)).clamp(-1, 1)
        omega = torch.acos(cos_sim)

        if omega.abs() < 1e-6:
            # Почти одинаковые: linear
            merged[key] = ((1 - t) * sd_a[key].float() + t * sd_b[key].float()
                          ).to(sd_a[key].dtype)
        else:
            sin_omega = torch.sin(omega)
            result = (torch.sin((1 - t) * omega) / sin_omega * a +
                     torch.sin(t * omega) / sin_omega * b)
            merged[key] = result.reshape(sd_a[key].shape).to(sd_a[key].dtype)

    return merged


def merge_models_ties(models, base_model=None, density=0.5):
    """
    TIES-Merging: Trim, Elect Sign, Merge.

    1. Trim: обнуляет малые изменения (по величине)
    2. Elect Sign: для каждого параметра выбирает знак большинства
    3. Merge: усредняет только параметры с выбранным знаком

    Ref: Yadav et al., "TIES-Merging" (2023)

    Args:
        models: list[nn.Module] — fine-tuned модели
        base_model: базовая модель (для вычисления дельт)
        density: доля параметров для сохранения (0-1)

    Returns:
        merged state_dict
    """
    if base_model is None:
        base_model = models[0]

    base_sd = base_model.state_dict()
    n = len(models)

    # 1. Вычисляем дельты
    deltas = []
    for m in models:
        delta = {}
        for key in base_sd:
            delta[key] = m.state_dict()[key].float() - base_sd[key].float()
        deltas.append(delta)

    merged = {}
    for key in base_sd:
        all_deltas = torch.stack([d[key] for d in deltas])  # (N, ...)

        # 1. Trim: обнуляем малые дельты
        magnitudes = all_deltas.abs()
        threshold = magnitudes.quantile(1.0 - density)
        trimmed = all_deltas.clone()
        trimmed[magnitudes < threshold] = 0

        # 2. Elect Sign: знак большинства
        sign_votes = trimmed.sign().sum(dim=0)
        elected_sign = sign_votes.sign()
        # При нуле (ничья) оставляем 0
        elected_sign[sign_votes == 0] = 0

        # 3. Merge: усредняем только дельты с выбранным знаком
        mask = trimmed.sign() == elected_sign.unsqueeze(0)
        masked_deltas = trimmed * mask.float()
        counts = mask.float().sum(dim=0).clamp(min=1)
        avg_delta = masked_deltas.sum(dim=0) / counts

        merged[key] = (base_sd[key].float() + avg_delta).to(base_sd[key].dtype)

    return merged

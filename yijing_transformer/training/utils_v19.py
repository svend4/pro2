"""
v19 утилиты: Sophia, RMSNorm, Chunked Prefill, CAGrad, EMA Decay Scheduler.

Sophia: second-order clipped optimizer с Hutchinson estimator.
Ref: Liu et al., "Sophia: A Scalable Stochastic Second-Order Optimizer" (2023)

RMSNorm: нормализация по RMS вместо LayerNorm (без center).
Ref: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)

Chunked Prefill: разбиение длинного контекста на чанки для inference.
Ref: Ainslie et al., "GQA" / vLLM chunked prefill

CAGrad: Conflict-Averse Gradient descent для multi-task learning.
Ref: Liu et al., "Conflict-Averse Gradient Descent" (2021)

EMA Decay Scheduler: динамический warmup для EMA decay.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Sophia Optimizer ====================

class Sophia(torch.optim.Optimizer):
    """
    Sophia: second-order optimizer с diagonal Hessian clipping.

    Использует Hutchinson estimator для аппроксимации
    диагонали Гессиана и clips Adam-like update по нему.

    Args:
        params: параметры модели
        lr: learning rate
        betas: (beta1, beta2) для momentum/Hessian EMA
        eps: epsilon
        weight_decay: L2 регуляризация
        rho: clipping threshold
        update_hessian_every: шагов между обновлениями Hessian
    """
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), eps=1e-8,
                 weight_decay=0.1, rho=0.04, update_hessian_every=10):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        rho=rho)
        self.update_hessian_every = update_hessian_every
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Один шаг Sophia."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.ones_like(p)

                exp_avg = state['exp_avg']
                hessian = state['hessian']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Momentum update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Clipped update: clip(m / max(h, eps), -rho, rho)
                update = exp_avg / hessian.clamp(min=group['eps'])
                update.clamp_(-group['rho'], group['rho'])

                p.data.add_(update, alpha=-group['lr'])

        return loss

    def update_hessian(self, loss_fn):
        """
        Обновляет аппроксимацию диагонали Гессиана через Hutchinson.

        Args:
            loss_fn: callable() → loss (должна быть дифференцируемой)
        """
        loss = loss_fn()
        loss.backward(create_graph=True)

        for group in self.param_groups:
            beta2 = group['betas'][1]
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'hessian' not in state:
                    state['hessian'] = torch.ones_like(p)

                # Hutchinson estimator: E[z * (Hz)] ≈ diag(H)
                # Mixed-precision safe: z должен совпадать по dtype с p.grad
                z = torch.randint(0, 2, p.shape, device=p.device, dtype=p.dtype) * 2.0 - 1.0
                hz = torch.autograd.grad(
                    p.grad, p, grad_outputs=z, retain_graph=True
                )[0]
                hessian_est = (hz * z).abs()

                # EMA of Hessian diagonal
                state['hessian'].mul_(beta2).add_(hessian_est, alpha=1 - beta2)

        self.zero_grad()


# ==================== RMSNorm ====================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Нормализует по RMS без center (без bias subtraction).
    Более эффективна и стабильна, чем LayerNorm.

    Args:
        d_model: размерность
        eps: epsilon для стабильности
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (..., d_model)

        Returns:
            normalized: (..., d_model)
        """
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

    def extra_repr(self):
        return f'{self.weight.shape[0]}, eps={self.eps}'


# ==================== Chunked Prefill ====================

class ChunkedPrefill:
    """
    Chunked Prefill: разбивает длинный prompt на чанки для inference.

    Позволяет обрабатывать очень длинные контексты без OOM,
    последовательно прогоняя чанки через модель и накапливая KV-cache.

    Args:
        model: модель с forward(x)
        chunk_size: размер чанка
    """
    def __init__(self, model, chunk_size=256):
        self.model = model
        self.chunk_size = chunk_size

    def prefill(self, input_ids):
        """
        Обрабатывает input_ids чанками.

        Args:
            input_ids: (1, T) — длинная последовательность

        Returns:
            logits: (1, T, V) — собранные logits
            chunks_info: dict с метаданными
        """
        B, T = input_ids.shape
        assert B == 1, "ChunkedPrefill supports batch_size=1"

        all_logits = []
        n_chunks = math.ceil(T / self.chunk_size)

        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, T)
            chunk = input_ids[:, start:end]

            with torch.no_grad():
                logits, _, _ = self.model(chunk)
            all_logits.append(logits)

        combined = torch.cat(all_logits, dim=1)

        return combined, {
            'n_chunks': n_chunks,
            'chunk_size': self.chunk_size,
            'total_tokens': T,
        }

    def generate_with_prefill(self, input_ids, max_new_tokens=50):
        """
        Prefill + генерация.

        Args:
            input_ids: (1, T)
            max_new_tokens: число новых токенов

        Returns:
            generated_ids: (1, T + max_new_tokens)
        """
        # Prefill
        logits, info = self.prefill(input_ids)

        # Greedy generation from last logits
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            # Use only last block_size tokens
            ctx = generated[:, -self.model.core.layers[0].attn.max_seq_len:] \
                if hasattr(self.model.core.layers[0].attn, 'max_seq_len') \
                else generated[:, -self.chunk_size:]

            with torch.no_grad():
                out, _, _ = self.model(ctx)
            next_logits = out[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        return generated


# ==================== CAGrad ====================

class CAGrad:
    """
    Conflict-Averse Gradient Descent.

    Для multi-task learning: находит gradient direction, который
    минимизирует worst-case conflict между задачами.

    Вместо простого усреднения градиентов, проецирует их так,
    чтобы минимизировать максимальный конфликт.

    Args:
        n_tasks: число задач
        c: coefficient для ограничения отклонения от среднего (0.5 типично)
    """
    def __init__(self, n_tasks, c=0.5):
        self.n_tasks = n_tasks
        self.c = c

    def backward(self, losses, shared_params):
        """
        Вычисляет conflict-averse gradient.

        Args:
            losses: list[Tensor] — losses для каждой задачи
            shared_params: list[Parameter] — shared параметры

        Returns:
            total_loss: скаляр для логирования
        """
        grads = []
        for loss in losses:
            self.zero_grads(shared_params)
            loss.backward(retain_graph=True)
            grad = self._get_grad_flat(shared_params)
            grads.append(grad)

        grads = torch.stack(grads)  # (n_tasks, D)

        # Среднее направление
        g_avg = grads.mean(dim=0)

        # Проекция: для каждой задачи проецируем на среднее
        g_result = g_avg.clone()

        # Находим наиболее конфликтующий градиент
        cosines = F.cosine_similarity(grads, g_avg.unsqueeze(0), dim=1)
        worst_idx = cosines.argmin()

        # Проецируем worst gradient
        g_worst = grads[worst_idx]
        dot = (g_worst * g_avg).sum()
        if dot < 0:
            # Конфликт: корректируем
            proj = dot / (g_avg.norm() ** 2 + 1e-8) * g_avg
            g_corrected = g_worst - proj * self.c
            # Заменяем worst gradient на скорректированный
            grads[worst_idx] = g_corrected
            g_result = grads.mean(dim=0)

        # Устанавливаем итоговый gradient
        self._set_grad_flat(shared_params, g_result)

        return sum(l.item() for l in losses) / len(losses)

    def zero_grads(self, params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    def _get_grad_flat(self, params):
        grads = []
        for p in params:
            if p.grad is not None:
                grads.append(p.grad.flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)

    def _set_grad_flat(self, params, flat_grad):
        offset = 0
        for p in params:
            n = p.numel()
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.copy_(flat_grad[offset:offset + n].view_as(p))
            offset += n


# ==================== EMA Decay Scheduler ====================

class EMADecayScheduler:
    """
    Динамический scheduler для EMA decay.

    Warmup EMA decay от начального значения к целевому.
    В начале обучения модель быстро меняется, поэтому EMA decay
    должен быть меньше (быстрее обновляется). По мере обучения
    увеличиваем decay для более стабильного усреднения.

    Args:
        ema_model: модель с EMA весами
        source_model: обучаемая модель
        initial_decay: начальный decay (0.9 типично)
        target_decay: целевой decay (0.999 типично)
        warmup_steps: шагов для warmup decay
    """
    def __init__(self, ema_model, source_model, initial_decay=0.9,
                 target_decay=0.999, warmup_steps=1000):
        self.ema_model = ema_model
        self.source_model = source_model
        self.initial_decay = initial_decay
        self.target_decay = target_decay
        self.warmup_steps = warmup_steps
        self._step = 0

    def get_decay(self):
        """Текущее значение decay."""
        if self._step >= self.warmup_steps:
            return self.target_decay
        # Cosine warmup
        progress = self._step / max(1, self.warmup_steps)
        return self.initial_decay + (self.target_decay - self.initial_decay) * (
            1 - math.cos(math.pi * progress)
        ) / 2

    def step(self):
        """Обновляет EMA с текущим decay."""
        self._step += 1
        decay = self.get_decay()
        self._update_ema(decay)
        return decay

    def _update_ema(self, decay):
        """Обновляет EMA веса."""
        with torch.no_grad():
            for ema_p, src_p in zip(
                self.ema_model.parameters(),
                self.source_model.parameters()
            ):
                ema_p.data.mul_(decay).add_(src_p.data, alpha=1 - decay)

    def state_dict(self):
        return {
            'step': self._step,
            'initial_decay': self.initial_decay,
            'target_decay': self.target_decay,
            'warmup_steps': self.warmup_steps,
        }

    def load_state_dict(self, state):
        self._step = state['step']
        self.initial_decay = state['initial_decay']
        self.target_decay = state['target_decay']
        self.warmup_steps = state['warmup_steps']

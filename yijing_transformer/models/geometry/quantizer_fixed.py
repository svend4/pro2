"""
quantizer_fixed.py — Корректная тернарная квантизация.

Проблема оригинала: STE zero-gradient trap при жёстких тритах.
Решение: Gumbel-Softmax с физически корректным расписанием
         (аналог алгоритма Метрополиса из meta/hexphys).

Проверка корректности:
    python yijing_transformer/models/geometry/quantizer_fixed.py --test
"""

import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryQuantizerFixed(nn.Module):
    """
    Тернарная квантизация {-1, 0, +1} через Gumbel-Softmax.

    Ключевое свойство: градиенты проходят НА ВСЁМ протяжении обучения.
    Тритам разрешено быть любым значением от -1 до +1 (мягко),
    постепенно твердея к дискретным значениям по мере снижения температуры.
    """

    def __init__(
        self,
        warmup_steps: int = 3000,
        start_temp: float = 1.0,
        end_temp: float = 0.05,
        schedule: str = 'cosine',  # 'cosine' | 'linear' | 'exponential'
    ):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.schedule = schedule

        self.register_buffer('step', torch.tensor(0, dtype=torch.long))

    def get_temperature(self) -> float:
        """Вычислить текущую температуру по расписанию."""
        progress = min(self.step.item() / max(self.warmup_steps, 1), 1.0)

        if self.schedule == 'cosine':
            # Медленный старт, медленный конец — аналог Метрополис-MCMC
            t = 1.0 - (1.0 - progress) * (1.0 - progress)
            factor = (1 - math.cos(math.pi * t)) / 2
        elif self.schedule == 'linear':
            factor = progress
        else:  # exponential
            factor = 1.0 - math.exp(-3 * progress)

        return self.start_temp + (self.end_temp - self.start_temp) * factor

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: любой формы, последнее измерение = количество архетипов

        Returns:
            trit: того же shape, значения в (-1, +1) при обучении,
                  в {-1, 0, +1} при инференсе
        """
        T = self.get_temperature()

        if self.training:
            # Три категории для каждого архетипа: {-1, 0, +1}
            logits_3 = torch.stack([
                -logits,
                torch.zeros_like(logits),
                logits,
            ], dim=-1)

            soft = F.gumbel_softmax(logits_3, tau=T, hard=False, dim=-1)
            # p(-1) * (-1) + p(0) * 0 + p(+1) * (+1)
            trit = soft[..., 2] - soft[..., 0]

            self.step += 1
            return trit
        else:
            # Инференс: жёсткая тернарная квантизация
            threshold = 0.3
            trit = torch.zeros_like(logits)
            trit[logits > threshold] = 1.0
            trit[logits < -threshold] = -1.0
            return trit

    def diagnostics(self) -> dict:
        return {
            'step': self.step.item(),
            'temperature': self.get_temperature(),
            'progress': min(self.step.item() / max(self.warmup_steps, 1), 1.0),
            'is_hard_phase': self.get_temperature() < 0.15,
        }


# ─── Самотест ─────────────────────────────────────────────────────────────────

def self_test():
    """
    Проверить что:
    1. Градиенты проходят при любой температуре
    2. Распределение тритов меняется с температурой
    3. К шагу warmup_steps триты дифференцированы
    """
    print("=" * 50)
    print("САМОТЕСТ TernaryQuantizerFixed")
    print("=" * 50)

    quantizer = TernaryQuantizerFixed(warmup_steps=100, start_temp=2.0, end_temp=0.05)

    # Тест 1: градиенты проходят
    logits = torch.randn(4, 16, requires_grad=True)
    trit = quantizer(logits)
    loss = trit.sum()
    loss.backward()

    grad_ok = logits.grad is not None and logits.grad.abs().max() > 0
    print(f"\n1. Градиенты {'ПРОХОДЯТ' if grad_ok else 'ЗАБЛОКИРОВАНЫ'}")
    print(f"   Max grad: {logits.grad.abs().max():.4f}")

    # Тест 2: температура снижается правильно
    temps = []
    q = TernaryQuantizerFixed(warmup_steps=200)
    dummy = torch.randn(2, 8)
    for _ in range(200):
        q(dummy)
        temps.append(q.get_temperature())

    temp_ok = temps[0] > 0.8 and temps[-1] < 0.2
    print(f"\n2. Температура {'снижается' if temp_ok else 'не снижается'}")
    print(f"   Старт: {temps[0]:.3f} | Шаг 100: {temps[99]:.3f} | Конец: {temps[-1]:.3f}")

    # Тест 3: к концу прогрева триты ближе к {-1, 0, +1}
    q_early = TernaryQuantizerFixed(warmup_steps=200)
    q_late = TernaryQuantizerFixed(warmup_steps=200)

    # Ускорить q_late до конца прогрева
    dummy = torch.randn(2, 8)
    for _ in range(190):
        q_late(dummy)

    with torch.no_grad():
        t_early = q_early(torch.randn(100, 16))
        t_late = q_late(torch.randn(100, 16))

    early_extreme = (t_early.abs() > 0.7).float().mean()
    late_extreme = (t_late.abs() > 0.7).float().mean()

    hard_ok = late_extreme > early_extreme
    print(f"\n3. Затвердевание {'работает' if hard_ok else 'не работает'}")
    print(f"   Ранние экстремальные триты: {early_extreme:.3f}")
    print(f"   Поздние экстремальные триты: {late_extreme:.3f}")

    all_ok = all([grad_ok, temp_ok, hard_ok])
    print(f"\n{'ВСЕ ТЕСТЫ ПРОЙДЕНЫ' if all_ok else 'ЕСТЬ ПРОБЛЕМЫ'}")
    return all_ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.test:
        self_test()
    else:
        # Быстрая демонстрация
        q = TernaryQuantizerFixed(warmup_steps=3000)
        x = torch.randn(4, 64)
        for step in [0, 500, 1500, 3000]:
            q.step.fill_(step)
            print(f"Step {step:4d}: T={q.get_temperature():.3f}, "
                  f"hard_phase={q.diagnostics()['is_hard_phase']}")

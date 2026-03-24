"""
CrossDomainAnalogy — модуль межэкспертных аналогий для NautilusMoME.

Идея (автор концепции — пользователь):
  Коллаж из разных доменов — не баг, а нереализованный потенциал.
  Как биофизика берёт формулы из физики для описания биологии,
  так и модель может использовать паттерны одного эксперта
  для описания данных другого.

  Пословицы и крылатые выражения — это "гуманитарные формулы":
  концентрированная мудрость в нескольких словах. Модель уже имеет
  информацию, но не умеет ей пользоваться. Нужны специальные
  микромодули для осмысленного кросс-доменного переноса.

Четырёхуровневая система (info3):
  Аналогии естественно возникают МЕЖДУ уровнями:
  - Formula↔Theorem (MATH↔HUMAN): математика психотипов
  - Archetype↔Algorithm (CODE↔RECON): паттерны → восстановление
  - Archetype↔Archetype (CODE↔SYSTEM): проектирование ↔ архитектура
  Эти связи ОРГАНИЧЕСКИ обнаруживаются через обучение, не задаются жёстко.

Архитектура:
  1. AnalogyMatrix — матрица C(6,2)=15 парных "мостиков" между 6 экспертами
  2. ProverbCondenser — сжатие последовательности в "формулу" (аналог пословицы)
  3. CrossDomainRouter — маршрутизатор, который активирует аналогии
     когда два эксперта имеют конфликтующие, но сильные сигналы

Встраивается между NautilusBridge и core_second:
  core_first → Router → Experts → Bridge → [CrossDomainAnalogy] → core_second
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


class ProverbCondenser(nn.Module):
    """Сжимает последовательность в 'формулу' — аналог пословицы.

    Пословица = концентрированная мудрость в минимуме слов.
    Этот модуль берёт вывод эксперта (B, T, D) и сжимает его
    в один вектор-формулу (B, D), который содержит "суть" домена.

    Механизм: attention-weighted pooling с обучаемым query.
    """

    def __init__(self, d_model):
        super().__init__()
        # "Вопрос к эксперту": обучаемый вектор, который спрашивает
        # "в чём суть того, что ты видишь?"
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, expert_output):
        """
        Args:
            expert_output: (B, T, D) — вывод одного эксперта
        Returns:
            formula: (B, 1, D) — сжатая "пословица"
        """
        B, T, D = expert_output.shape

        q = self.query.expand(B, -1, -1)          # (B, 1, D)
        k = self.key_proj(expert_output)            # (B, T, D)
        v = self.value_proj(expert_output)           # (B, T, D)

        # Attention: query спрашивает "в чём суть?"
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, 1, T)
        attn = F.softmax(attn, dim=-1)

        formula = torch.bmm(attn, v)  # (B, 1, D)
        return formula


class AnalogyPair(nn.Module):
    """Один мостик-аналогия между двумя доменами.

    Пример: MATH↔HUMAN — находит аналогии между
    математическими структурами и гуманитарными паттернами.

    Как биофизика: берёт "формулу" из домена A и проецирует
    её в пространство домена B, создавая кросс-доменный инсайт.
    """

    def __init__(self, d_model, d_analogy=None):
        super().__init__()
        d_analogy = d_analogy or d_model // 2

        # A → общее пространство аналогии
        self.proj_a = nn.Linear(d_model, d_analogy)
        # B → общее пространство аналогии
        self.proj_b = nn.Linear(d_model, d_analogy)

        # Измерение "силы аналогии" (насколько домены похожи в данном контексте)
        self.similarity_gate = nn.Sequential(
            nn.Linear(d_analogy * 2, d_analogy),
            nn.GELU(),
            nn.Linear(d_analogy, 1),
            nn.Sigmoid(),
        )

        # Синтез: создаёт новое знание на пересечении доменов
        self.synthesis = nn.Sequential(
            nn.Linear(d_analogy * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Гейт для стабильности (начинаем с малого влияния)
        self.gate = nn.Parameter(torch.tensor(0.01))

    def forward(self, formula_a, formula_b):
        """
        Args:
            formula_a: (B, 1, D) — "пословица" эксперта A
            formula_b: (B, 1, D) — "пословица" эксперта B
        Returns:
            insight: (B, 1, D) — кросс-доменный инсайт
            strength: (B, 1, 1) — сила аналогии
        """
        # Проецируем в общее пространство аналогий
        a = self.proj_a(formula_a)  # (B, 1, d_analogy)
        b = self.proj_b(formula_b)  # (B, 1, d_analogy)

        # Измеряем силу аналогии
        combined = torch.cat([a, b], dim=-1)  # (B, 1, d_analogy*2)
        strength = self.similarity_gate(combined)  # (B, 1, 1)

        # Синтезируем новое знание
        insight = self.synthesis(combined) * self.gate  # (B, 1, D)

        return insight, strength


class CrossDomainAnalogy(nn.Module):
    """Модуль межэкспертных аналогий.

    Для 6 экспертов создаёт полную матрицу 6×6=36 взаимодействий:
      - 15 верхних пар (A→B undirected, исходная реализация)
      - 15 нижних пар (B→A reverse direction, новое)
      - 6 диагональных (A→A self-analogy: как домен аналогичен себе через время)

    Всего: 36 направленных взаимодействий = полный граф (Aut(Q6) orbits).

    Ключевые направленные аналогии:
      MATH→CODE: формальная структура → алгоритм
      CODE→MATH: алгоритм → формальная верификация
      HUMAN→SYSTEM: этические принципы → системные ограничения
      SYSTEM→HUMAN: системная архитектура → социальные структуры
      MATH→MATH: самоотсылка (Гёдель) — математика порождает математику
    """

    # Default expert names — can be overridden via constructor.
    # These are organic labels describing what the router learns to recognize,
    # not hard domain boundaries. The router discovers specialization from content.
    EXPERT_NAMES = ['MATH', 'CODE', 'HUMAN', 'SYSTEM', 'RECON', 'INFO']

    def __init__(self, d_model, n_experts=6, expert_names=None,
                 d_analogy=None, threshold=0.3):
        super().__init__()
        self.n_experts = n_experts
        self.threshold = threshold
        if expert_names is not None:
            self.EXPERT_NAMES = expert_names
        names = self.EXPERT_NAMES[:n_experts]

        # ProverbCondenser для каждого эксперта
        self.condensers = nn.ModuleDict({
            name: ProverbCondenser(d_model) for name in names
        })

        # AnalogyPair для всех 36 клеток матрицы 6×6
        # Ключ: "{from}_{to}" — направленная аналогия (A→B ≠ B→A)
        self.pairs = nn.ModuleDict()
        self.pair_keys = []  # [(key, i, j)] где i=source, j=target
        for i in range(n_experts):
            for j in range(n_experts):
                key = f"{names[i]}_{names[j]}"
                self.pairs[key] = AnalogyPair(d_model, d_analogy)
                self.pair_keys.append((key, i, j))

        # Финальная проекция: объединяет все инсайты
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        # Общий гейт: насколько сильно аналогии влияют на результат
        self.analogy_gate = nn.Parameter(torch.tensor(0.05))

    def forward(self, bridge_output, expert_outputs, expert_weights):
        """
        Args:
            bridge_output: (B, T, D) — вывод NautilusBridge
            expert_outputs: list of (B, T, D) — выходы экспертов
            expert_weights: (B, T, n_experts) — веса маршрутизации
        Returns:
            output: (B, T, D) — bridge_output + аналоговая добавка
            analogy_info: dict — отладочная информация
        """
        B, T, D = bridge_output.shape
        names = self.EXPERT_NAMES[:self.n_experts]

        # Шаг 1: Сжимаем каждый эксперт в "пословицу"
        formulas = {}
        for i, name in enumerate(names):
            # Взвешиваем выход эксперта его маршрутизационным весом
            w = expert_weights[:, :, i:i+1]  # (B, T, 1)
            weighted_output = expert_outputs[i] * w
            formulas[name] = self.condensers[name](weighted_output)  # (B, 1, D)

        # Шаг 2: Находим аналогии между парами
        total_insight = torch.zeros(B, 1, D, device=bridge_output.device)
        analogy_strengths = {}
        active_count = 0

        for key, i, j in self.pair_keys:
            insight, strength = self.pairs[key](formulas[names[i]], formulas[names[j]])

            avg_strength = strength.mean().item()
            analogy_strengths[key] = avg_strength

            # Используем аналогию только если она достаточно сильная
            if avg_strength > self.threshold:
                total_insight = total_insight + insight * strength
                active_count += 1

        # Шаг 3: Проецируем и расширяем инсайт на всю последовательность
        if active_count > 0:
            # Нормализуем по количеству активных аналогий
            total_insight = total_insight / max(active_count, 1)
            # Проецируем
            projected = self.output_proj(total_insight)  # (B, 1, D)
            # Расширяем на все позиции
            analogy_addition = projected.expand(B, T, D) * self.analogy_gate
            output = bridge_output + analogy_addition
        else:
            output = bridge_output
            analogy_addition = None

        analogy_info = {
            'strengths': analogy_strengths,
            'active_pairs': active_count,
            'total_pairs': len(self.pair_keys),
            'gate_value': self.analogy_gate.item(),
        }

        return output, analogy_info


# ==================== Демонстрация ====================

def demo_cross_domain_analogy():
    """Показывает как работает CrossDomainAnalogy."""
    d_model = 128
    n_experts = 6
    B, T = 2, 32

    module = CrossDomainAnalogy(d_model, n_experts)
    total_params = sum(p.numel() for p in module.parameters())
    print(f"CrossDomainAnalogy parameters: {total_params:,}")
    print(f"Number of analogy pairs: {len(module.pair_keys)} (6×6=36 directed)")
    # Show diagonal (self-analogies) and off-diagonal separately
    diag = [(k, i, j) for k, i, j in module.pair_keys if i == j]
    off  = [(k, i, j) for k, i, j in module.pair_keys if i != j]
    print(f"  Diagonal (self): {[k for k, _, _ in diag]}")
    print(f"  Off-diagonal directed: {len(off)} pairs")
    print()

    # Симулируем
    bridge_output = torch.randn(B, T, d_model)
    expert_outputs = [torch.randn(B, T, d_model) for _ in range(n_experts)]
    expert_weights = torch.softmax(torch.randn(B, T, n_experts), dim=-1)

    output, info = module(bridge_output, expert_outputs, expert_weights)

    print(f"Input shape:  {bridge_output.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Active analogy pairs: {info['active_pairs']}/{info['total_pairs']}")
    print(f"Gate value: {info['gate_value']:.4f}")
    print()
    print("Analogy strengths:")
    for pair, strength in sorted(info['strengths'].items(), key=lambda x: -x[1]):
        bar = '█' * int(strength * 40)
        active = " ← ACTIVE" if strength > 0.3 else ""
        print(f"  {pair:15s}  {strength:.3f}  {bar}{active}")

    return module, info


if __name__ == '__main__':
    demo_cross_domain_analogy()

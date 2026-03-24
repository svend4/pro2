"""
KasatkinQ6Router — Касаткин как РОУТЕР, не как attention bias (Шаг 5).

Ключевое отличие от v59: CubicAttentionBias добавлял Касаткина как bias
к attention score → не помогало. Правильная роль — координатная система
для маршрутизации 6 экспертов через 3D-проекцию.

Каждый из 6 экспертов = одна из 6 осей Q6-гиперкуба.
Запрос проецируется в 3D → ближайшая ось = выбранный эксперт.

Преимущество: визуализируемость. Можно видеть attention как потоки
в кубе — именно то, о чём писал Касаткин.

Протокол теста (Шаг 7):
    - Минимум 3000 шагов, реальные данные
    - Успех: routing_confidence > 15% при PPL не хуже LeanYiJing baseline
    - Провал до этого — недействителен
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Имена доменов (из CONCEPTUAL_STAGE.md)
DOMAIN_NAMES = ["GEO", "HYDRO", "PYRO", "AERO", "COSMO", "NOOS"]
DOMAIN_ALT   = ["CODE", "RECON", "SYSTEM", "MATH", "HUMAN", "INFO"]


class KasatkinQ6Router(nn.Module):
    """Маршрутизация 6 экспертов через 3D-проекцию Касаткина.

    Архитектура:
        x (B, T, d_model)
        → proj_3d (B, T, 3)  — координаты в кубе
        → cosine similarity с 6 осями куба
        → softmax → routing_weights (B, T, 6)

    6 осей куба = 6 направлений = 6 линий гексаграммы = 6 экспертов:
        ±X, ±Y, ±Z → 6 различных вычислительных путей

    Args:
        d_model: размерность входных векторов
        n_experts: ДОЛЖНО быть 6 (= числу осей Q6-куба)
        routing_temperature: температура softmax (ниже → резче выбор)
        use_learned_axes: разрешить обучение осей (по умолчанию фиксированы)
    """

    # Имена доменов для визуализации и ксерокс-теста
    EXPERT_NAMES = DOMAIN_ALT  # ['CODE', 'RECON', 'SYSTEM', 'MATH', 'HUMAN', 'INFO']

    # 6 осей куба: ±X, ±Y, ±Z
    _EXPERT_AXES = torch.tensor([
        [1., 0., 0.],    # +X = CODE (GEO)
        [-1., 0., 0.],   # -X = RECON (HYDRO)
        [0., 1., 0.],    # +Y = SYSTEM (PYRO)
        [0., -1., 0.],   # -Y = MATH (AERO)
        [0., 0., 1.],    # +Z = HUMAN (COSMO)
        [0., 0., -1.],   # -Z = INFO (NOOS)
    ])

    def __init__(
        self,
        d_model: int = 128,
        n_experts: int = 6,
        routing_temperature: float = 0.5,
        use_learned_axes: bool = False,
    ):
        super().__init__()
        if n_experts != 6:
            raise ValueError(
                f"KasatkinQ6Router работает ровно с 6 экспертами "
                f"(= 6 осей куба), получено n_experts={n_experts}"
            )
        self.d_model = d_model
        self.n_experts = n_experts
        self.routing_temperature = routing_temperature

        # Проектор d_model → 3D (обучаемый)
        self.proj_3d = nn.Linear(d_model, 3, bias=False)

        if use_learned_axes:
            self.expert_axes = nn.Parameter(self._EXPERT_AXES.clone())
        else:
            # 6 осей (не обучаются — фиксированная геометрия)
            self.register_buffer('expert_axes', self._EXPERT_AXES)

        # Инициализация: ортогональная проекция
        nn.init.orthogonal_(self.proj_3d.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Вычисляет мягкие routing weights через Q6-геометрию.

        Args:
            x: (B, T, d_model) — входные представления

        Returns:
            routing_weights: (B, T, n_experts) — мягкие веса экспертов
        """
        # Проекция в 3D-куб Касаткина
        coords_3d = self.proj_3d(x)  # (B, T, 3)

        # Нормализация для косинусного сходства
        coords_norm = F.normalize(coords_3d, dim=-1)  # (B, T, 3)

        # Сходство с каждой из 6 осей
        axes = F.normalize(self.expert_axes, dim=-1)  # (6, 3)
        axes_exp = axes.view(1, 1, 6, 3)
        coords_exp = coords_norm.unsqueeze(2)  # (B, T, 1, 3)
        similarities = (coords_exp * axes_exp).sum(dim=-1)  # (B, T, 6)

        # Мягкие routing weights
        routing_weights = F.softmax(
            similarities / self.routing_temperature,
            dim=-1,
        )

        return routing_weights

    def route_sequence(
        self,
        x: torch.Tensor,
        expert_modules: list,
    ) -> torch.Tensor:
        """Применяет MoE-шаг: routing_weights × expert_outputs.

        Args:
            x: (B, T, d_model)
            expert_modules: список из 6 nn.Module

        Returns:
            output: (B, T, d_model) — взвешенная смесь экспертов
        """
        assert len(expert_modules) == self.n_experts

        routing_weights = self.forward(x)  # (B, T, 6)

        expert_outputs = torch.stack(
            [expert(x) for expert in expert_modules],
            dim=-1,
        )  # (B, T, d_model, 6)

        output = (
            expert_outputs * routing_weights.unsqueeze(2)
        ).sum(dim=-1)  # (B, T, d_model)

        return output

    def get_routing_confidence(self, x: torch.Tensor) -> float:
        """Метрика уверенности роутера.

        Цель: > 15% (т.е. max weight > 1/6 + 0.15 ≈ 0.32).

        Args:
            x: (B, T, d_model)

        Returns:
            confidence: скаляр — среднее (max_weight - 1/n_experts)
        """
        weights = self.forward(x)  # (B, T, 6)
        max_weights = weights.max(dim=-1).values  # (B, T)
        confidence = (max_weights - 1.0 / self.n_experts).mean().item()
        return confidence

    def get_routing_weights_for_text(
        self,
        text: str,
        embed_fn,
        device: str = "cpu",
    ) -> dict:
        """Утилита для ксерокс-теста: routing для строки текста.

        Args:
            text: входная строка
            embed_fn: функция text → (1, T, d_model) tensor

        Returns:
            {domain_name: weight} — нормализованные веса по доменам
        """
        with torch.no_grad():
            x = embed_fn(text).to(device)
            weights = self.forward(x)  # (1, T, 6)
            avg_weights = weights.mean(dim=(0, 1))  # (6,)

        return {
            name: avg_weights[i].item()
            for i, name in enumerate(self.EXPERT_NAMES)
        }

    def hex_label(self, x: torch.Tensor) -> list:
        """
        Вернуть читаемую метку домена для каждого токена.
        Используется для интерпретации и ксерокс-теста.

        Args:
            x: (T, d_model) или (1, T, d_model)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        with torch.no_grad():
            routing = self.forward(x)  # (1, T, 6)
            active = routing.argmax(dim=-1)  # (1, T)

        labels = []
        for t in range(active.shape[-1]):
            idx = active[0, t].item()
            coords = self.proj_3d(x[0, t]).tolist()
            labels.append({
                'token_pos': t,
                'expert':    idx,
                'domain':    DOMAIN_NAMES[idx],
                'alt_name':  DOMAIN_ALT[idx],
                'coord_3d':  [round(c, 3) for c in coords],
            })
        return labels

    def visualize_routing(self, x: torch.Tensor):
        """Возвращает 3D-координаты для визуализации потоков в кубе.

        Args:
            x: (B, T, d_model)

        Returns:
            coords: (T, 3) numpy array (первый батч)
        """
        with torch.no_grad():
            coords = self.proj_3d(x[0])  # (T, 3)
        return coords.cpu().numpy()


class Q6ExpertBank(nn.Module):
    """
    Банк из 6 экспертов с Q6-маршрутизацией.
    Заменяет стандартный MoE с softmax-routing.
    """

    def __init__(self, d_model: int, d_ffn: int, n_experts: int = 6):
        super().__init__()
        self.router = KasatkinQ6Router(d_model, n_experts=n_experts)

        # 6 FFN-экспертов (по одному на домен)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.GELU(),
                nn.Linear(d_ffn, d_model),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model) — взвешенная смесь выходов экспертов
        """
        routing = self.router(x)  # (B, T, 6)

        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )  # (B, T, d_model, 6)

        routing_expanded = routing.unsqueeze(-2)  # (B, T, 1, 6)
        out = (expert_outputs * routing_expanded).sum(dim=-1)
        return out


if __name__ == '__main__':
    print("Тест KasatkinQ6Router...")
    d = 64
    router = KasatkinQ6Router(d_model=d, routing_temperature=0.3)

    x = torch.randn(2, 16, d)
    weights = router.forward(x)
    print(f"  Routing weights shape: {weights.shape}")   # (2, 16, 6)
    print(f"  Weights sum (must~1):  {weights.sum(dim=-1).mean():.4f}")

    conf = router.get_routing_confidence(x)
    print(f"  Routing confidence:   {conf:.4f}  (цель > 0.15)")

    labels = router.hex_label(x[0])
    print(f"  Hex-label t=0: {labels[0]['domain']}/{labels[0]['alt_name']} {labels[0]['coord_3d']}")

    print("\nТест Q6ExpertBank...")
    bank = Q6ExpertBank(d_model=d, d_ffn=d * 4)
    out = bank(x)
    print(f"  Output shape: {out.shape}")   # (2, 16, 64)

    print("\n✅ Тест завершён")

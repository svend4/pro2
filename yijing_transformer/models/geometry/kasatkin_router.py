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
    """

    # Имена доменов для визуализации и ксерокс-теста
    EXPERT_NAMES = ['CODE', 'RECON', 'SYSTEM', 'MATH', 'HUMAN', 'INFO']

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
        # expert_axes: (6, 3) → unsqueeze → (1, 1, 6, 3)
        axes = self.expert_axes.view(1, 1, 6, 3)
        coords_exp = coords_norm.unsqueeze(2)  # (B, T, 1, 3)
        similarities = (coords_exp * axes).sum(dim=-1)  # (B, T, 6)

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
            expert_modules: список из 6 nn.Module (каждый: (B, T, d_model) → (B, T, d_model))

        Returns:
            output: (B, T, d_model) — взвешенная смесь экспертов
        """
        assert len(expert_modules) == self.n_experts

        routing_weights = self.forward(x)  # (B, T, 6)

        # Вычисляем вывод каждого эксперта
        expert_outputs = torch.stack(
            [expert(x) for expert in expert_modules],
            dim=-1,
        )  # (B, T, d_model, 6)

        # Взвешенная сумма
        output = (
            expert_outputs * routing_weights.unsqueeze(2)
        ).sum(dim=-1)  # (B, T, d_model)

        return output

    def get_routing_confidence(self, x: torch.Tensor) -> float:
        """Метрика уверенности роутера.

        Высокая уверенность = одна ось доминирует → хорошая специализация.
        Цель: > 15% (т.е. max weight > 1/6 + 0.15 = ~0.32).

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
            device: устройство

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

    def visualize_routing(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Возвращает 3D-координаты для визуализации потоков в кубе.

        Именно та «визуализируемость» о которой писал Касаткин.

        Args:
            x: (B, T, d_model)

        Returns:
            coords: (T, 3) numpy array (первый батч)
        """
        with torch.no_grad():
            coords = self.proj_3d(x[0])  # (T, 3)
        return coords.cpu().numpy()


if __name__ == '__main__':
    router = KasatkinQ6Router(d_model=128)
    x = torch.randn(2, 16, 128)
    weights = router.forward(x)
    print(f"Routing weights shape: {weights.shape}")  # (2, 16, 6)
    print(f"Weights sum: {weights.sum(dim=-1).mean():.4f}")  # ~1.0
    conf = router.get_routing_confidence(x)
    print(f"Routing confidence: {conf:.4f}")
    coords = router.visualize_routing(x)
    print(f"3D coords shape: {coords.shape}")  # (16, 3)

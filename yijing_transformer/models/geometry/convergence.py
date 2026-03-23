"""
Convergence Bridge: конвергентная иерархия глифов и токенов.

Два потока абстракции сходятся в «золотой середине»:
- GlyphComposer: Q6 примитивы → составные сигилы (снизу вверх)
- TokenAbstractor: конкретные токены → 64 архетипа-кластера (сверху вниз)
- ConvergenceLayer: cross-attention слияние обоих потоков

Биологическая аналогия:
    Глифы: Вид → Род → Семейство → Отряд (↑ композиция)
    Токены: Вид → Род → Семейство → Отряд (↑ абстрагирование)
    Встреча на уровне «Отряд» — общее пространство R^d_model

64 кластера TokenAbstractor = 64 гексаграммы Q6.
Если корреляция выучивается — эмпирическое подтверждение архетипов И-Цзин.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GlyphComposer(nn.Module):
    """Компонует последовательность Q6-вершин в составные сигилы.

    Иерархия композиции:
        Уровень 0: Глиф — 1 вершина Q6 (6 бит)
        Уровень 1: Ребро — 2 соседних вершины (центр + направление)
        Уровень 2: Грань — окно из k вершин → спектральные признаки подграфа
        Уровень 3: Сигил — полный подграф с adjacency и спектром Лапласиана

    Каждый уровень проецируется в d_model, давая иерархическое представление.

    Args:
        d_model: размерность модели
        window_size: размер окна для формирования сигилов (4-8)
        stride: шаг скользящего окна
        n_compose_layers: число слоёв self-attention для композиции
    """

    def __init__(self, d_model: int, window_size: int = 4,
                 stride: int = 2, n_compose_layers: int = 1):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.stride = stride

        # Проекция одного Q6-глифа в d_model
        self.glyph_proj = nn.Linear(6, d_model, bias=False)

        # Проекция ребра (центр 6D + направление 6D = 12D)
        self.edge_proj = nn.Linear(12, d_model, bias=False)

        # Спектральные признаки подграфа из k вершин:
        # - centroid (6D)
        # - spread (1D)
        # - adjacency eigenvalues (window_size D) — спектр Лапласиана
        spectral_dim = 6 + 1 + window_size
        self.spectral_proj = nn.Linear(spectral_dim, d_model, bias=False)

        # Self-attention для рафинирования сигилов
        if n_compose_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=min(4, d_model // 16 or 1),
                dim_feedforward=d_model * 2,
                dropout=0.1, batch_first=True,
                norm_first=True,
            )
            self.composer = nn.TransformerEncoder(layer, num_layers=n_compose_layers)
        else:
            self.composer = None

        # Выходная проекция + norm
        self.out_norm = nn.LayerNorm(d_model)

        # Learnable scale (начинаем с малого вклада)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def _compute_edges(self, vertices: Tensor) -> Tensor:
        """Вычисляет рёбра (пары соседних вершин).

        Args:
            vertices: (batch, seq_len, 6) Q6 вершины

        Returns:
            edges: (batch, seq_len-1, 12) — [center, direction] для каждого ребра
        """
        # Центр = среднее двух соседних вершин
        center = (vertices[:, :-1] + vertices[:, 1:]) / 2.0  # (B, T-1, 6)
        # Направление = разность
        direction = vertices[:, 1:] - vertices[:, :-1]  # (B, T-1, 6)
        return torch.cat([center, direction], dim=-1)  # (B, T-1, 12)

    def _compute_sigil_features(self, vertices: Tensor) -> Tensor:
        """Вычисляет спектральные признаки для скользящих окон.

        Для каждого окна из k вершин:
        1. Centroid (6D) — центр масс
        2. Spread (1D) — стандартное отклонение
        3. Laplacian spectrum (kD) — собственные значения графового Лапласиана

        Args:
            vertices: (batch, seq_len, 6) Q6 вершины

        Returns:
            features: (batch, n_windows, spectral_dim)
        """
        B, T, D = vertices.shape
        k = self.window_size

        # Число окон
        if T < k:
            # Если последовательность короче окна — pad
            pad = k - T
            vertices = F.pad(vertices, (0, 0, 0, pad))
            T = k

        n_windows = max(1, (T - k) // self.stride + 1)
        features = []

        for w in range(n_windows):
            start = w * self.stride
            end = start + k
            window = vertices[:, start:end, :]  # (B, k, 6)

            # 1. Centroid
            centroid = window.mean(dim=1)  # (B, 6)

            # 2. Spread (среднее L2 расстояние от центроида)
            spread = ((window - centroid.unsqueeze(1)) ** 2).sum(dim=-1).mean(dim=1, keepdim=True).add(1e-8).sqrt()  # (B, 1)

            # 3. Спектр Лапласиана подграфа
            # Adjacency: a_ij ≈ σ(gain·(2 - hamming)) — soft порог вокруг hamming=2.
            # В Q6 с {-1,+1}: Hamming = (6 - dot(v_i, v_j)) / 2
            # Sigmoid вместо boolean (hamming<=2).float() сохраняет градиенты:
            # d(adj)/d(dots) ≠ 0, поэтому eigenvalues Лапласиана обучаемы.
            dots = torch.bmm(window, window.transpose(1, 2))  # (B, k, k)
            hamming = (6.0 - dots) / 2.0
            adj = torch.sigmoid(4.0 * (2.0 - hamming))  # soft adjacency, grad-friendly
            # Убираем self-loops для Лапласиана
            adj = adj * (1.0 - torch.eye(k, device=adj.device).unsqueeze(0))

            # Degree matrix
            degree = adj.sum(dim=-1)  # (B, k)
            # Laplacian L = D - A
            L = torch.diag_embed(degree) - adj  # (B, k, k)

            # Собственные значения (отсортированные)
            eigenvalues = torch.linalg.eigvalsh(L)  # (B, k)

            # Конкатенируем признаки
            feat = torch.cat([centroid, spread, eigenvalues], dim=-1)  # (B, spectral_dim)
            features.append(feat)

        return torch.stack(features, dim=1)  # (B, n_windows, spectral_dim)

    def forward(self, glyph_vertices: Tensor) -> Tensor:
        """Компонует Q6-вершины в сигилы.

        Args:
            glyph_vertices: (batch, seq_len, 6) — координаты Q6 {-1, +1}

        Returns:
            sigil_embeddings: (batch, n_sigils, d_model)
        """
        B, T, _ = glyph_vertices.shape

        # Уровень 0: проекция глифов
        glyph_emb = self.glyph_proj(glyph_vertices)  # (B, T, d_model)

        # Уровень 1: рёбра
        if T >= 2:
            edges = self._compute_edges(glyph_vertices)  # (B, T-1, 12)
            edge_emb = self.edge_proj(edges)  # (B, T-1, d_model)
        else:
            edge_emb = glyph_emb

        # Уровень 2: спектральные признаки сигилов
        sigil_features = self._compute_sigil_features(glyph_vertices)  # (B, n_windows, spectral_dim)
        sigil_emb = self.spectral_proj(sigil_features)  # (B, n_windows, d_model)

        # Объединяем все уровни через pooled edges
        # Агрегируем edges в те же окна что и sigils
        n_sigils = sigil_emb.shape[1]
        edge_pooled = []
        for w in range(n_sigils):
            start = w * self.stride
            end = min(start + self.window_size, edge_emb.shape[1])
            if start < edge_emb.shape[1]:
                edge_pooled.append(edge_emb[:, start:end].mean(dim=1))
            else:
                edge_pooled.append(torch.zeros(B, self.d_model, device=edge_emb.device, dtype=edge_emb.dtype))
        edge_pooled = torch.stack(edge_pooled, dim=1)  # (B, n_sigils, d_model)

        # Glyph-level pooling (тот же windowing)
        glyph_pooled = []
        for w in range(n_sigils):
            start = w * self.stride
            end = min(start + self.window_size, glyph_emb.shape[1])
            glyph_pooled.append(glyph_emb[:, start:end].mean(dim=1))
        glyph_pooled = torch.stack(glyph_pooled, dim=1)  # (B, n_sigils, d_model)

        # Иерархическое слияние: глифы + рёбра + спектр
        composed = glyph_pooled + edge_pooled + sigil_emb  # (B, n_sigils, d_model)

        # Self-attention рафинирование
        if self.composer is not None:
            composed = self.composer(composed)

        return self.out_norm(composed) * self.scale


class TokenAbstractor(nn.Module):
    """Обобщает конкретные токены до уровня 64 архетипов-кластеров.

    64 кластера = 64 гексаграммы — не совпадение:
    каждый кластер потенциально соответствует одной из 64 вершин Q6.

    Механизм:
    1. Soft k-means: каждый токен → мягкое присваивание к кластеру
    2. Центры кластеров = learnable параметры в R^d_model
    3. Temperature annealing: начинаем мягко → жёсткие кластеры

    Дополнительно:
    - Кластерные центры инициализируются на вершинах Q6
    - Измеряется mutual information между кластерами и гексаграммами

    Args:
        d_model: размерность модели
        n_clusters: число кластеров (по умолчанию 64 = число гексаграмм)
        init_temperature: начальная температура softmax
    """

    def __init__(self, d_model: int, n_clusters: int = 64,
                 init_temperature: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.n_clusters = n_clusters

        # Центры кластеров (learnable)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, d_model) * 0.02)

        # Температура для soft assignment (learnable)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temperature)))

        # Проекция для вычисления расстояний (bottleneck)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)

        # Выходная проекция: кластерное представление → d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = nn.LayerNorm(d_model)

        # Q6 координаты кластеров (для анализа корреляции с гексаграммами)
        self._init_hexagram_anchors()

        # Learnable scale
        self.scale = nn.Parameter(torch.tensor(0.1))

    def _init_hexagram_anchors(self):
        """Инициализирует Q6 якори для 64 кластеров = 64 гексаграммы."""
        anchors = []
        for i in range(64):
            vertex = tuple(2 * ((i >> (5 - b)) & 1) - 1 for b in range(6))
            anchors.append(vertex)
        # Register as buffer (не обучается, но сохраняется)
        self.register_buffer(
            'hexagram_anchors',
            torch.tensor(anchors, dtype=torch.float32)  # (64, 6)
        )

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def forward(self, token_embeddings: Tensor) -> tuple:
        """Абстрагирует токены до уровня кластеров.

        Args:
            token_embeddings: (batch, seq_len, d_model)

        Returns:
            abstract: (batch, seq_len, d_model) — кластерное представление
            assignments: (batch, seq_len, n_clusters) — мягкие присваивания
        """
        # Проецируем для вычисления расстояний
        queries = self.query_proj(token_embeddings)  # (B, T, d_model)

        # Cosine similarity вместо L2 — более стабильно
        queries_norm = F.normalize(queries, dim=-1)
        centers_norm = F.normalize(self.cluster_centers, dim=-1)

        # Сходство: (B, T, n_clusters)
        similarity = torch.matmul(queries_norm, centers_norm.T)

        # Soft assignment с температурой
        assignments = F.softmax(similarity / self.temperature, dim=-1)  # (B, T, 64)

        # Абстрактное представление = взвешенная сумма центров
        abstract = torch.matmul(assignments, self.cluster_centers)  # (B, T, d_model)

        # Выходная проекция
        abstract = self.out_proj(abstract)
        abstract = self.out_norm(abstract) * self.scale

        return abstract, assignments

    def cluster_hexagram_correlation(self) -> Tensor:
        """Измеряет корреляцию между выученными кластерами и гексаграммами Q6.

        Проецирует центры кластеров в 6D и сравнивает с гексаграммами.
        Высокая корреляция = эмпирическое подтверждение архетипов И-Цзин.

        Returns:
            correlation: скаляр [0, 1] — нормализованная корреляция
        """
        # Берём первые 6 компонент центров как «скрытые Q6 координаты»
        # (или можно добавить отдельную проекцию d_model → 6)
        if self.d_model >= 6:
            centers_q6 = self.cluster_centers[:, :6]  # (64, 6)
        else:
            return self.cluster_centers.new_tensor(0.0)

        # Sign → бинаризация
        centers_binary = centers_q6.sign()  # (64, 6) в {-1, +1}

        # Для каждого кластера находим ближайшую гексаграмму
        # dots: (64, 64) — скалярные произведения
        dots = torch.matmul(centers_binary, self.hexagram_anchors.T)  # (64, 64)

        # Максимальное сходство для каждого кластера
        max_similarity = dots.max(dim=1).values  # (64,)

        # Нормализуем: максимум = 6 (полное совпадение)
        correlation = max_similarity.mean() / 6.0  # [0, 1]

        return correlation


class ConvergenceLayer(nn.Module):
    """Золотая середина: слияние сигилов (снизу) и абстрактных токенов (сверху).

    Cross-attention между двумя потоками:
    - Сигилы attend к абстрактным токенам (что значит эта геометрия?)
    - Абстрактные токены attend к сигилам (какая геометрия у этого смысла?)
    - Gated merge → единое представление

    Args:
        d_model: размерность модели
        n_heads: число голов cross-attention
        dropout: dropout rate
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Cross-attention: сигилы ← абстрактные токены
        self.cross_attn_s2a = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention: абстрактные токены ← сигилы
        self.cross_attn_a2s = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # LayerNorms
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        # Gated merge: [s2a, a2s] → gate → merged
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # Final projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, sigils: Tensor, abstracts: Tensor) -> Tensor:
        """Сливает два потока.

        Args:
            sigils: (batch, n_sigils, d_model) — от GlyphComposer
            abstracts: (batch, seq_len, d_model) — от TokenAbstractor

        Returns:
            merged: (batch, seq_len, d_model) — конвергентное представление
        """
        # Нормализация
        s = self.norm_s(sigils)
        a = self.norm_a(abstracts)

        # Cross-attention в обе стороны
        # Сигилы спрашивают у абстрактных токенов: "что я значу?"
        s2a, _ = self.cross_attn_s2a(query=s, key=a, value=a)  # (B, n_sigils, d_model)

        # Абстрактные токены спрашивают у сигилов: "какая у меня геометрия?"
        a2s, _ = self.cross_attn_a2s(query=a, key=s, value=s)  # (B, seq_len, d_model)

        # Приводим s2a к размерности seq_len через interpolation
        if s2a.shape[1] != a2s.shape[1]:
            s2a = F.interpolate(
                s2a.transpose(1, 2),
                size=a2s.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # (B, seq_len, d_model)

        # Gated merge
        g = self.gate(torch.cat([s2a, a2s], dim=-1))  # (B, seq_len, d_model) → sigmoid
        merged = g * s2a + (1 - g) * a2s  # (B, seq_len, d_model)

        return self.norm_out(self.out_proj(merged))


class ConvergenceBridge(nn.Module):
    """Полный мост конвергентной иерархии.

    Объединяет все три компонента:
    1. GlyphComposer: текст → Q6 вершины → сигилы
    2. TokenAbstractor: token embeddings → 64 кластера
    3. ConvergenceLayer: сигилы × кластеры → конвергентное представление

    Выход добавляется к основному потоку трансформера через residual + scale.

    Args:
        d_model: размерность модели
        n_clusters: число кластеров (64 = число гексаграмм)
        window_size: размер окна для GlyphComposer
        stride: шаг окна
        n_compose_layers: число слоёв self-attention в GlyphComposer
        n_heads: число голов cross-attention в ConvergenceLayer
    """

    def __init__(self, d_model: int, n_clusters: int = 64,
                 window_size: int = 4, stride: int = 2,
                 n_compose_layers: int = 1, n_heads: int = 4):
        super().__init__()

        self.glyph_composer = GlyphComposer(
            d_model=d_model,
            window_size=window_size,
            stride=stride,
            n_compose_layers=n_compose_layers,
        )
        self.token_abstractor = TokenAbstractor(
            d_model=d_model,
            n_clusters=n_clusters,
        )
        self.convergence = ConvergenceLayer(
            d_model=d_model,
            n_heads=n_heads,
        )

        # Общий scale для residual connection
        self.bridge_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, token_embeddings: Tensor,
                glyph_vertices: Tensor) -> tuple:
        """Полный forward pass конвергентного моста.

        Args:
            token_embeddings: (batch, seq_len, d_model) — из стандартного embedding
            glyph_vertices: (batch, seq_len, 6) — Q6 координаты от GlyphTokenizer

        Returns:
            enriched: (batch, seq_len, d_model) — обогащённое представление
            info: dict с диагностикой:
                - assignments: мягкие присваивания к кластерам
                - correlation: корреляция кластеров с гексаграммами
        """
        # Поток снизу: глифы → сигилы
        sigils = self.glyph_composer(glyph_vertices)  # (B, n_sigils, d_model)

        # Поток сверху: токены → кластеры
        abstracts, assignments = self.token_abstractor(token_embeddings)  # (B, T, d_model)

        # Встреча в середине
        converged = self.convergence(sigils, abstracts)  # (B, T, d_model)

        # Residual: добавляем к основному потоку
        enriched = token_embeddings + self.bridge_scale * converged

        info = {
            'assignments': assignments,
            'correlation': self.token_abstractor.cluster_hexagram_correlation(),
            'n_sigils': sigils.shape[1],
        }

        return enriched, info

    def get_convergence_loss(self, assignments: Tensor) -> Tensor:
        """Вспомогательный loss для обучения конвергенции.

        Два компонента:
        1. Entropy regularization: поощряем чёткие (но не одноточечные) кластеры
        2. Balance loss: все кластеры должны использоваться примерно одинаково

        Args:
            assignments: (batch, seq_len, n_clusters) — мягкие присваивания

        Returns:
            loss: скалярный loss
        """
        # 1. Entropy: не слишком размазанный, не слишком жёсткий
        # Целевая entropy ≈ log(8) — каждый токен "видит" ~8 кластеров
        entropy = -(assignments * (assignments + 1e-10).log()).sum(dim=-1)  # (B, T)
        target_entropy = math.log(8.0)
        entropy_loss = ((entropy - target_entropy) ** 2).mean()

        # 2. Balance: среднее использование кластеров ≈ uniform
        avg_usage = assignments.mean(dim=(0, 1))  # (n_clusters,)
        target_usage = 1.0 / assignments.shape[-1]
        balance_loss = ((avg_usage - target_usage) ** 2).sum() * assignments.shape[-1]

        return 0.1 * entropy_loss + 0.1 * balance_loss


class MatrixGrammar(nn.Module):
    """Матричная грамматика сигилов — 2D представление вместо 1D цепочки.

    Вдохновлено Atamiri (Гусман де Рохас): «Аймара — это матрица,
    и её синтаксис определён в массиве». Синтаксис = не дерево и не
    последовательность, а двумерный массив.

    Архитектура:
        1D сигилы → reshape в 2D матрицу (rows × cols)
        Строки = синтаксические роли (агент, действие, объект, локация...)
        Столбцы = семантические слоты (тип, модификатор, связь...)
        2D self-attention по строкам и столбцам (axial attention)
        → Развёртка обратно в 1D

    Это обобщение: стандартный 1D attention видит только цепочку,
    MatrixGrammar видит таблицу, где отношения по двум осям имеют
    разный смысл.

    Args:
        d_model: размерность модели
        n_rows: число строк матрицы (синтаксические роли)
        n_cols: число столбцов (семантические слоты)
        n_heads: число голов в axial attention
    """

    def __init__(self, d_model: int, n_rows: int = 8, n_cols: int = 8,
                 n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_slots = n_rows * n_cols  # 64 = число гексаграмм!

        # Проекция из 1D потока в 2D слоты
        self.slot_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable slot embeddings (какая роль у каждого слота)
        self.row_emb = nn.Parameter(torch.randn(n_rows, d_model // 2) * 0.02)
        self.col_emb = nn.Parameter(torch.randn(n_cols, d_model // 2) * 0.02)

        # Axial attention: по строкам (синтаксис)
        self.row_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        # Axial attention: по столбцам (семантика)
        self.col_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )

        # Norms
        self.norm_row = nn.LayerNorm(d_model)
        self.norm_col = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        # Gated readout: матрица → 1D
        self.readout = nn.Linear(d_model, d_model, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def _assign_to_slots(self, x: Tensor) -> Tensor:
        """Мягко присваивает seq_len токенов к n_slots слотам матрицы.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            slots: (batch, n_slots, d_model)
        """
        B, T, D = x.shape

        # Создаём slot queries из row + col embeddings
        # row_emb: (n_rows, D//2), col_emb: (n_cols, D//2)
        # Outer product → (n_rows, n_cols, D)
        row_exp = self.row_emb.unsqueeze(1).expand(-1, self.n_cols, -1)  # (R, C, D//2)
        col_exp = self.col_emb.unsqueeze(0).expand(self.n_rows, -1, -1)  # (R, C, D//2)
        slot_queries = torch.cat([row_exp, col_exp], dim=-1)  # (R, C, D)
        slot_queries = slot_queries.reshape(self.n_slots, D)  # (n_slots, D)

        # Cross-attention: slots attend к входным токенам
        # Similarity: (B, n_slots, T)
        x_proj = self.slot_proj(x)  # (B, T, D)
        sim = torch.matmul(
            slot_queries.unsqueeze(0).expand(B, -1, -1),  # (B, n_slots, D)
            x_proj.transpose(1, 2)  # (B, D, T)
        ) / math.sqrt(D)

        # Soft assignment
        weights = F.softmax(sim, dim=-1)  # (B, n_slots, T)

        # Weighted sum: каждый слот = weighted pool входных токенов
        slots = torch.bmm(weights, x)  # (B, n_slots, D)

        return slots

    def _axial_attention(self, matrix: Tensor) -> Tensor:
        """Axial attention: сначала по строкам, потом по столбцам.

        Args:
            matrix: (batch, n_rows, n_cols, d_model)

        Returns:
            matrix: (batch, n_rows, n_cols, d_model)
        """
        B, R, C, D = matrix.shape

        # Row attention: каждая строка = последовательность из C токенов
        rows = matrix.reshape(B * R, C, D)
        rows = self.norm_row(rows)
        rows_attn, _ = self.row_attn(rows, rows, rows)
        rows = rows + rows_attn
        matrix = rows.reshape(B, R, C, D)

        # Col attention: каждый столбец = последовательность из R токенов
        cols = matrix.permute(0, 2, 1, 3).reshape(B * C, R, D)
        cols = self.norm_col(cols)
        cols_attn, _ = self.col_attn(cols, cols, cols)
        cols = cols + cols_attn
        matrix = cols.reshape(B, C, R, D).permute(0, 2, 1, 3)

        return matrix

    def forward(self, x: Tensor) -> Tensor:
        """Матричная грамматика: 1D → 2D матрица → axial attention → 1D.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            enriched: (batch, seq_len, d_model) — обогащённое представление
        """
        B, T, D = x.shape

        # 1. Присваиваем токены к слотам 2D матрицы
        slots = self._assign_to_slots(x)  # (B, n_slots, D)

        # 2. Reshape в 2D матрицу
        matrix = slots.reshape(B, self.n_rows, self.n_cols, D)

        # 3. Axial attention
        matrix = self._axial_attention(matrix)

        # 4. Развёртка обратно и readout к seq_len
        slots_processed = matrix.reshape(B, self.n_slots, D)

        # 5. Обратная проекция: slots → tokens
        # Используем те же slot queries для обратного маппинга
        row_exp = self.row_emb.unsqueeze(1).expand(-1, self.n_cols, -1)
        col_exp = self.col_emb.unsqueeze(0).expand(self.n_rows, -1, -1)
        slot_keys = torch.cat([row_exp, col_exp], dim=-1).reshape(self.n_slots, D)

        # Similarity: tokens attend к обработанным слотам
        x_proj = self.slot_proj(x)  # reuse projection
        sim = torch.matmul(
            x_proj,  # (B, T, D)
            slot_keys.T  # (D, n_slots)
        ) / math.sqrt(D)
        weights = F.softmax(sim, dim=-1)  # (B, T, n_slots)

        # Reconstruct: каждый токен = weighted sum обработанных слотов
        readout = torch.bmm(weights, slots_processed)  # (B, T, D)

        return self.norm_out(self.readout(readout)) * self.scale

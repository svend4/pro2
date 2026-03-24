"""
Визуализация И-Цзин геометрии и обученных параметров.

Создаёт набор графиков:
1. 3D куб триграмм с подписями
2. PCA-проекция 64 гексаграмм (6D → 2D/3D)
3. Матрица расстояний между триграммами
4. Визуализация квантизации: как точки проецируются на вершины
5. Обученные 变卦 вероятности
6. Адаптивные температуры по слоям
7. Loss curves (если доступны)

Использование:
    python scripts/visualize_geometry.py
    python scripts/visualize_geometry.py --checkpoint checkpoints/checkpoint_step_2000.pt
    python scripts/visualize_geometry.py --save-dir plots/
"""

import sys
import os
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.geometry import generate_trigrams, generate_hexagrams, FactoredYiJingQuantizer

# И-Цзин символика
TRIGRAM_NAMES = {
    (1, 1, 1): '☰ Цянь (Небо)',
    (1, 1, -1): '☱ Дуй (Озеро)',
    (1, -1, 1): '☲ Ли (Огонь)',
    (1, -1, -1): '☳ Чжэнь (Гром)',
    (-1, 1, 1): '☴ Сюнь (Ветер)',
    (-1, 1, -1): '☵ Кань (Вода)',
    (-1, -1, 1): '☶ Гэнь (Гора)',
    (-1, -1, -1): '☷ Кунь (Земля)',
}


def plot_trigram_cube(save_path=None):
    """3D куб триграмм с рёбрами и подписями."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available, skipping trigram cube plot")
        return

    tri = generate_trigrams()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Рёбра куба (соединяем вершины на расстоянии Хэмминга 1)
    for i in range(8):
        for j in range(i + 1, 8):
            diff = (tri[i] - tri[j]).abs().sum().item()
            if diff == 2.0:  # Хэмминг 1
                ax.plot3D(
                    [tri[i, 0].item(), tri[j, 0].item()],
                    [tri[i, 1].item(), tri[j, 1].item()],
                    [tri[i, 2].item(), tri[j, 2].item()],
                    'b-', alpha=0.3, linewidth=0.8
                )

    # Вершины
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71',
              '#1abc9c', '#3498db', '#9b59b6', '#34495e']
    for i in range(8):
        key = tuple(int(v) for v in tri[i].tolist())
        name = TRIGRAM_NAMES.get(key, f'T{i}')
        ax.scatter(tri[i, 0], tri[i, 1], tri[i, 2],
                   s=200, c=colors[i], zorder=5, edgecolors='k')
        ax.text(tri[i, 0].item() * 1.15, tri[i, 1].item() * 1.15,
                tri[i, 2].item() * 1.15, name, fontsize=8, ha='center')

    ax.set_xlabel('Линия 1 (Инь/Ян)')
    ax.set_ylabel('Линия 2')
    ax.set_zlabel('Линия 3')
    ax.set_title('8 Триграмм — вершины куба {-1, +1}³')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.savefig('trigram_cube.png', dpi=150, bbox_inches='tight')
        print("Saved: trigram_cube.png")
    plt.close()


def plot_hexagram_pca(save_path=None):
    """PCA-проекция 64 гексаграмм из 6D в 2D."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping hexagram PCA plot")
        return

    hex = generate_hexagrams()  # (64, 6)

    # PCA через SVD
    hex_centered = hex - hex.mean(dim=0)
    U, S, V = torch.svd(hex_centered)
    projected = hex_centered @ V[:, :2]  # (64, 2)

    # Раскраска по верхней триграмме (первые 3 координаты)
    upper_idx = []
    tri = generate_trigrams()
    for i in range(64):
        for j in range(8):
            if torch.allclose(hex[i, :3], tri[j]):
                upper_idx.append(j)
                break

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.Set1(range(8))

    for j in range(8):
        mask = [i for i, u in enumerate(upper_idx) if u == j]
        key = tuple(int(v) for v in tri[j].tolist())
        name = TRIGRAM_NAMES.get(key, f'T{j}')
        ax.scatter(
            projected[mask, 0], projected[mask, 1],
            c=[colors[j]], s=100, label=f'Верх: {name}',
            edgecolors='k', linewidths=0.5
        )

    # Подписи: номера гексаграмм
    for i in range(64):
        ax.annotate(str(i), (projected[i, 0].item(), projected[i, 1].item()),
                    fontsize=6, ha='center', va='center', alpha=0.6)

    ax.set_xlabel(f'PC1 (σ={S[0]:.2f})')
    ax.set_ylabel(f'PC2 (σ={S[1]:.2f})')
    ax.set_title('64 Гексаграммы — PCA проекция из {-1,+1}⁶')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    path = save_path or 'hexagram_pca.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_distance_matrix(save_path=None):
    """Матрица расстояний между триграммами (Хэмминг)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    tri = generate_trigrams()
    dists = torch.cdist(tri, tri)
    hamming = (dists ** 2 / 4).int()

    labels = []
    for i in range(8):
        key = tuple(int(v) for v in tri[i].tolist())
        labels.append(TRIGRAM_NAMES.get(key, f'T{i}').split(' ')[0])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(hamming.float(), cmap='YlOrRd', vmin=0, vmax=3)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    for i in range(8):
        for j in range(8):
            ax.text(j, i, str(hamming[i, j].item()),
                    ha='center', va='center', fontsize=11,
                    color='white' if hamming[i, j] >= 2 else 'black')

    ax.set_title('Расстояние Хэмминга между триграммами')
    fig.colorbar(im, ax=ax, label='Хэмминг')

    path = save_path or 'trigram_distances.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_quantization_demo(save_path=None):
    """Визуализация квантизации: случайные точки → вершины гиперкуба."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    q = FactoredYiJingQuantizer(temp=0.1)
    tri = generate_trigrams()

    # Случайные точки в 3D (одна триграмма)
    z = torch.randn(200, 3) * 0.8
    z_hard = q._soft_quantize(z)

    fig = plt.figure(figsize=(14, 6))

    # До квантизации
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(z[:, 0], z[:, 1], z[:, 2], c='blue', alpha=0.3, s=20)
    for i in range(8):
        ax1.scatter(*tri[i].tolist(), c='red', s=200, marker='*', edgecolors='k')
    ax1.set_title('До квантизации')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_zlim(-2, 2)

    # После квантизации
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(z_hard[:, 0], z_hard[:, 1], z_hard[:, 2], c='green', alpha=0.3, s=20)
    for i in range(8):
        ax2.scatter(*tri[i].tolist(), c='red', s=200, marker='*', edgecolors='k')
    ax2.set_title('После soft-квантизации (temp=0.1)')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_zlim(-2, 2)

    path = save_path or 'quantization_demo.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_model_analysis(checkpoint_path, save_dir='.'):
    """Визуализация обученных параметров модели."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = ckpt['model_state_dict']

    # Собираем hex_scale, head_scales, temp, bian_gua по слоям
    hex_scales = []
    head_scales = []
    temps = []
    bian_gua_scales = []
    change_probs_all = []

    i = 0
    while True:
        key = f'core.layers.{i}.hex_scale'
        if key not in state:
            break
        hex_scales.append(state[key].item())
        head_scales.append(state[f'core.layers.{i}.attn.head_scales'].abs().mean().item())
        temp_key = f'core.layers.{i}.quantizer.log_temp'
        if temp_key in state:
            temps.append(state[temp_key].exp().clamp(0.01, 5.0).item())
        bg_key = f'core.layers.{i}.bian_gua.scale'
        if bg_key in state:
            bian_gua_scales.append(state[bg_key].item())
            logits = state[f'core.layers.{i}.bian_gua.change_logits']
            change_probs_all.append(torch.sigmoid(logits).tolist())
        i += 1

    n_layers = i
    if n_layers == 0:
        print("No layers found in checkpoint")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Hex scales per layer
    axes[0, 0].bar(range(n_layers), hex_scales, color='coral')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('hex_scale')
    axes[0, 0].set_title('Гексаграммный вклад по слоям')
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Head scales
    axes[0, 1].bar(range(n_layers), head_scales, color='steelblue')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('mean |head_scale|')
    axes[0, 1].set_title('Триграммный bias в attention')

    # Temperatures
    if temps:
        axes[1, 0].bar(range(n_layers), temps, color='mediumseagreen')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Temperature')
        axes[1, 0].set_title('Адаптивная температура квантизации')
        axes[1, 0].axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='init=0.3')
        axes[1, 0].legend()

    # BianGua change probabilities
    if change_probs_all:
        yao_names = ['爻1', '爻2', '爻3', '爻4', '爻5', '爻6']
        import numpy as np
        data = np.array(change_probs_all)  # (n_layers, 6)
        im = axes[1, 1].imshow(data.T, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Линия (爻)')
        axes[1, 1].set_yticks(range(6))
        axes[1, 1].set_yticklabels(yao_names)
        axes[1, 1].set_title('变卦: вероятности изменения линий')
        fig.colorbar(im, ax=axes[1, 1])

    plt.suptitle(f'YiJing-Transformer Analysis ({n_layers} layers)', fontsize=14)
    plt.tight_layout()

    path = os.path.join(save_dir, 'model_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='YiJing Geometry Visualization')
    parser.add_argument('--save-dir', type=str, default='plots')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 50)
    print("YiJing Geometry Visualization")
    print("=" * 50)

    plot_trigram_cube(os.path.join(args.save_dir, 'trigram_cube.png'))
    plot_hexagram_pca(os.path.join(args.save_dir, 'hexagram_pca.png'))
    plot_distance_matrix(os.path.join(args.save_dir, 'trigram_distances.png'))
    plot_quantization_demo(os.path.join(args.save_dir, 'quantization_demo.png'))

    if args.checkpoint:
        plot_model_analysis(args.checkpoint, args.save_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()

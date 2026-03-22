"""
LeanYiJing: только то, что доказанно работает (Шаг 3).

Модель использует только HeisenbergAttention и FlowerOfLifeGAT —
два источника с подтверждёнными результатами из ablation v59.

Цель: установить официальный baseline PPL < 1.07.
Любое добавление компонента должно улучшать этот результат.

Протокол запуска:
    python -m yijing_transformer.models.lean_model
    Минимум 3000 шагов, реальные данные (не синтетический WikiText).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry.attention import HeisenbergAttention, FlowerOfLifeGAT


class LeanYiJingBlock(nn.Module):
    """Один блок LeanYiJing: Heisenberg + FlowerOfLifeGAT с learnable gate.

    Args:
        d_model: размерность модели
        n_heads: число голов attention
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.heisenberg = HeisenbergAttention(d_model, n_heads)
        self.flower_gat = FlowerOfLifeGAT(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Learnable gate: модель сама учится балансировать между источниками
        # Инициализируем ближе к 0.5 (нейтральный баланс)
        self.gate = nn.Parameter(torch.tensor(0.0))

        # FFN (стандартный)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            x: (B, T, d_model)
        """
        # Heisenberg attention
        h_attn = self.heisenberg(self.norm1(x))

        # FlowerOfLifeGAT
        h_gat = self.flower_gat(self.norm2(x))

        # Learnable gate (sigmoid → [0, 1])
        alpha = torch.sigmoid(self.gate)
        x = x + alpha * h_attn + (1 - alpha) * h_gat

        # FFN
        x = x + self.ffn(self.norm_ffn(x))

        return x


class LeanYiJingGPT(nn.Module):
    """Минимальная YiJing-GPT с двумя доказанными источниками.

    Параметры по умолчанию соответствуют конфигурации ablation v59:
    d_model=128, n_layers=4, n_heads=4, block_size=256.

    Args:
        vocab_size: размер словаря
        d_model: размерность модели
        n_layers: число блоков
        n_heads: число голов attention
        block_size: максимальная длина последовательности
        dropout: dropout (0.0 = без dropout для воспроизводимости)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        block_size: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            LeanYiJingBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (стандартная практика GPT)
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        """Стандартная GPT-инициализация."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple:
        """
        Args:
            idx: (B, T) — индексы токенов
            targets: (B, T) — целевые индексы для LM loss

        Returns:
            (logits, loss) — loss=None если targets не переданы
        """
        B, T = idx.shape
        assert T <= self.block_size, (
            f"Sequence length {T} > block_size {self.block_size}"
        )

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.embed(idx) + self.pos_embed(pos))

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Авторегрессивная генерация.

        Args:
            idx: (B, T) — начальный контекст
            max_new_tokens: сколько токенов сгенерировать
            temperature: температура семплирования
            top_k: если задан — top-k семплирование

        Returns:
            idx: (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    # Быстрая проверка
    model = LeanYiJingGPT()
    print(f"Параметры: {model.num_parameters:,}")

    x = torch.randint(0, 256, (2, 64))
    targets = torch.randint(0, 256, (2, 64))
    logits, loss = model(x, targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Gate значения после инициализации
    for i, block in enumerate(model.blocks):
        alpha = torch.sigmoid(block.gate).item()
        print(f"Block {i} gate (alpha): {alpha:.3f}")

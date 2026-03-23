"""
training_config.py — Централизованная конфигурация гиперпараметров обучения.

Использование:
    from training_config import TC

    optimizer = torch.optim.Adam(model.parameters(), lr=TC.lr_stage0)
    for step in range(TC.stage0_steps): ...
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainingConfig:
    # ------------------------------------------------------------------
    # Параметры модели (должны совпадать с Variant3Config в self_train_common)
    # ------------------------------------------------------------------
    vocab_size: int = 256
    block_size: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_mult: int = 4
    hamming_lambda: float = 0.15
    uncertainty_budget: float = 0.25
    dropout: float = 0.05

    # ------------------------------------------------------------------
    # Скорости обучения по стадиям
    # ------------------------------------------------------------------
    lr_stage0: float = 3e-4   # стадия 0: предобучение на корпусе
    lr_stage1: float = 1e-4   # стадия 1: само-обучение с RAG
    lr_stage2: float = 5e-5   # стадия 2: доводка с domain-loss
    lr_finetune: float = 2e-5 # тонкая настройка (v2, доп. цикл)
    lr_micro: float = 1e-5    # микро-обучение на паузах (самообучение v1)

    # ------------------------------------------------------------------
    # Шаги и батчи по стадиям
    # ------------------------------------------------------------------
    stage0_steps: int = 500   # v3: scarab; v1 использует 400
    stage1_steps: int = 600   # v1; v2/v3 используют 500
    stage2_steps: int = 400
    stage0_batch: int = 16
    stage1_batch: int = 8
    stage2_batch: int = 8

    # ------------------------------------------------------------------
    # Качество и буфер RAG
    # ------------------------------------------------------------------
    quality_threshold: float = 0.55   # минимальная оценка TextQualityFilter
    rag_sim_threshold: float = 0.90   # косинусное сходство для RAG-дедупликации

    # ------------------------------------------------------------------
    # LCI / Kirchhoff / Pipeline
    # ------------------------------------------------------------------
    lci_threshold: float = 2.8        # порог LCI для адаптивного LR в pipeline
    lci_resonance_target: float = 3.141592653589793  # π — цель Σ(gate_k × LCI_k)
    lci_resonance_window: float = 0.5  # |LCI - π| < 0.5 → резонанс
    turbine_lci_loss: float = 0.1     # вес Kirchhoff-штрафа в loss (pipeline)
    lci_loss_lambda: float = 0.0      # lci_loss_lambda по умолчанию (turbine)

    # ------------------------------------------------------------------
    # Параметры турбины / nautilus
    # ------------------------------------------------------------------
    turbine_temperature: float = 1.4  # начальная температура смягчения
    turbine_steps_per_expert: int = 20
    turbine_cycles: int = 8           # число циклов turbine в pipeline
    turbine_spe_pipeline: int = 8     # steps_per_expert для pipeline-вызова
    pipeline_step_scale: float = 0.4  # масштаб шагов в pipeline

    # Шаги кольца nautilus_4agent (пропорционально размерам орбит Q6)
    nautilus_steps_meta: int = 10     # орбиты 0,6 — полюса (2 вершины)
    nautilus_steps_abstract: int = 26  # орбиты 4,5 — Yang (21 вершина)
    nautilus_steps_dynamic: int = 25  # орбита 3 — экватор (20 вершин)
    nautilus_steps_concrete: int = 26  # орбиты 1,2 — Yin (21 вершина)

    # ------------------------------------------------------------------
    # v3: дополнительные параметры
    # ------------------------------------------------------------------
    v3_n_cycles: int = 4
    v3_temperature: float = 1.1       # температура в цикле scarab v3


# Единственный экземпляр, используется как: from training_config import TC
TC = TrainingConfig()

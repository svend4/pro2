"""
Обучение YiJing-Transformer с поддержкой:
- Cosine LR schedule с warmup
- Gradient accumulation
- Валидация
- Resume из чекпоинтов
- WandB / TensorBoard логирование
- Синтетические данные (для быстрых экспериментов) или TinyStories

Использование:
    # Синтетические данные (быстрый тест)
    python training/train.py --synthetic --steps 500

    # TinyStories
    python training/train.py --steps 50000

    # С WandB
    python training/train.py --wandb --run-name "yijing-v2-rope-swiglu"

    # Resume
    python training/train.py --resume checkpoints/checkpoint_step_2000.pt
"""

import os
import sys
import math
import time
import argparse
from dataclasses import asdict

import torch
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT
from models.baseline import VanillaGPT

# Bridge: подключение всех утилит из utils_v12..v52
from training.bridge import TrainingBridge


# ==================== УТИЛИТЫ ====================

def get_lr(step, cfg):
    """Cosine learning rate с warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def generate_synthetic_batch(batch_size, block_size, vocab_size, device):
    """Синтетические данные с паттернами (лучше чем чистый random)."""
    x = torch.randint(0, vocab_size, (batch_size, block_size + 1), device=device)
    # Добавляем простые паттерны: копирование части последовательности
    for i in range(batch_size):
        if i % 3 == 0:
            src_start = torch.randint(0, block_size // 2, (1,)).item()
            src_len = min(block_size // 4, block_size - src_start)
            dst_start = src_start + block_size // 2
            if dst_start + src_len <= block_size + 1:
                x[i, dst_start:dst_start + src_len] = x[i, src_start:src_start + src_len]
    return x[:, :-1], x[:, 1:]


# ==================== ЛОГИРОВАНИЕ ====================

class Logger:
    """Универсальный логгер: stdout + WandB + TensorBoard."""
    def __init__(self, cfg, args):
        self.use_wandb = cfg.use_wandb or getattr(args, 'wandb', False)
        self.use_tb = cfg.use_tensorboard or getattr(args, 'tensorboard', False)
        self.wandb_run = None
        self.tb_writer = None

        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=cfg.project_name,
                    name=cfg.run_name or args.run_name,
                    config=asdict(cfg),
                )
            except ImportError:
                print("wandb not installed, falling back to stdout")
                self.use_wandb = False

        if self.use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = getattr(args, 'tb_dir', 'runs')
                self.tb_writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("tensorboard not installed, falling back to stdout")
                self.use_tb = False

    def log(self, metrics: dict, step: int):
        parts = [f"Step {step}:"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(" ".join(parts))

        if self.use_wandb and self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=step)

        if self.use_tb and self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, step)

    def close(self):
        if self.use_wandb and self.wandb_run is not None:
            import wandb
            wandb.finish()
        if self.use_tb and self.tb_writer is not None:
            self.tb_writer.close()


# ==================== ВАЛИДАЦИЯ ====================

@torch.no_grad()
def estimate_val_loss(model, cfg, device, num_batches=20, data_fn=None):
    model.eval()
    losses = []
    for _ in range(num_batches):
        domain_ids = None
        if data_fn:
            batch = data_fn()
            if len(batch) == 3:
                xb, yb, domain_ids = batch
            else:
                xb, yb = batch
        else:
            xb, yb = generate_synthetic_batch(
                cfg.batch_size, cfg.block_size, cfg.vocab_size, device
            )
        _, loss, _ = model(xb, yb, domain_ids=domain_ids)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float('nan')


# ==================== АНАЛИЗ ГЕОМЕТРИИ ====================

def measure_hex_contribution(model):
    """Измеряет реальный вклад геометрических компонентов."""
    contributions = {}
    for i, layer in enumerate(model.core.layers):
        info = {
            'hex_scale': layer.hex_scale.item(),
            'mean_head_scale': layer.attn.head_scales.data.abs().mean().item(),
        }
        if hasattr(layer.quantizer, 'log_temp'):
            info['temp'] = layer.quantizer.current_temp.item()
        if layer.bian_gua is not None:
            info['bian_gua_scale'] = layer.bian_gua.scale.item()
            info['change_probs'] = [round(p, 3) for p in
                                     torch.sigmoid(layer.bian_gua.change_logits).tolist()]
        contributions[f'layer_{i}'] = info
    return contributions


# ==================== ОБУЧЕНИЕ ====================

def train(args):
    device = torch.device(args.device)

    # Предзагрузка корпуса svend4 ДО создания модели — нужен vocab_size
    _svend4_corpus = None
    if getattr(args, 'svend4', None):
        try:
            from data_utils.svend4_dataset import Svend4Corpus
            _domains = args.svend4_domains.split(",") if getattr(args, 'svend4_domains', None) else None
            _svend4_corpus = Svend4Corpus.from_directory(
                args.svend4, block_size=args.block_size, domains=_domains
            )
            args.vocab_size = _svend4_corpus.get_vocab_size()
        except Exception as _e:
            print(f"Не удалось загрузить svend4 корпус ({_e}), используем synthetic")
            args.svend4 = None

    # Glyph tokenizer подразумевает convergence bridge
    use_glyph = getattr(args, 'glyph_tokenizer', False)

    cfg = YiJingConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        block_size=args.block_size,
        batch_size=args.batch_size,
        warmup_steps=min(args.warmup, args.steps),
        total_steps=args.steps,
        use_rope=args.rope,
        use_swiglu=args.swiglu,
        use_bian_gua=args.bian_gua,
        use_hex_moe=args.moe,
        use_domain_moe=getattr(args, 'domain_moe', False),
        domain_moe_n_experts=getattr(args, 'domain_moe_experts', 6),
        domain_moe_top_k=getattr(args, 'domain_moe_top_k', 2),
        domain_supervision_weight=getattr(args, 'domain_supervision_weight', 0.1),
        adaptive_temp=args.adaptive_temp,
        use_wandb=args.wandb,
        use_tensorboard=args.tensorboard,
        run_name=args.run_name,
        use_amp=getattr(args, 'amp', False),
        n_kv_heads=getattr(args, 'gqa_heads', None),
        sliding_window=getattr(args, 'sliding_window', None),
        use_convergence_bridge=use_glyph or getattr(args, 'convergence_bridge', False),
        use_glyph_tokenizer=use_glyph,
        use_nautilus=getattr(args, 'nautilus', False),
        nautilus_mode=getattr(args, 'nautilus_mode', 'sequential'),
        nautilus_init_scale=getattr(args, 'nautilus_init_scale', 0.01),
        nautilus_warmup_steps=getattr(args, 'nautilus_warmup_steps', 2000),
        nautilus_chambers=getattr(args, 'nautilus_chambers', 'all'),
    )

    logger = Logger(cfg, args)

    if args.model == 'yijing':
        model = YiJingGPT(cfg).to(device)
    else:
        model = VanillaGPT(cfg).to(device)

    total_params, hex_params = model.count_parameters()
    print(f"Model: {args.model}")
    print(f"Parameters: {total_params:,}")
    if hex_params > 0:
        print(f"YiJing-specific: {hex_params:,} ({100*hex_params/total_params:.2f}%)")
    print(f"Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, block_size={cfg.block_size}")
    print(f"Features: RoPE={cfg.use_rope}, SwiGLU={cfg.use_swiglu}, "
          f"BianGua={cfg.use_bian_gua}, MoE={cfg.use_hex_moe}, "
          f"AdaptiveTemp={cfg.adaptive_temp}, "
          f"GlyphTokenizer={cfg.use_glyph_tokenizer}, "
          f"Nautilus={cfg.use_nautilus}")
    print()

    # === Bridge: пробуждение спящих утилит из v12..v52 ===
    bridge = TrainingBridge(model, cfg)

    # Оптимизатор: через bridge (поддержка Sophia, LAMB, Lion, SAM, Lookahead)
    optimizer_type = getattr(args, 'optimizer', 'adamw')
    optimizer_wrapper = getattr(args, 'optimizer_wrapper', None)
    optimizer = bridge.build_optimizer(
        optimizer_type=optimizer_type,
        wrapper=optimizer_wrapper,
        llrd_factor=getattr(args, 'llrd_factor', 1.0),
    )

    # Scheduler: через bridge (поддержка WSD, Cosine Restarts, Curriculum)
    scheduler_type = getattr(args, 'scheduler', 'cosine')
    scheduler = bridge.build_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
    )

    # Регуляризация: через bridge (Z-Loss, AGC, Mixup, Label Smoothing, etc.)
    regularization = bridge.build_regularization()

    # Мониторинг: через bridge (Loss Spike, Grokking, Gradient Flow, etc.)
    monitor = bridge.build_monitor(verbose=True)

    # Data pipeline: через bridge (Packing, BPE Dropout, Freq Weighting)
    data_pipeline = bridge.build_data_pipeline()

    # Model surgery: через bridge (µP init, pruning, freezing)
    surgeon = bridge.build_surgeon()
    if getattr(args, 'mup_init', False):
        surgeon.apply_mup_init()
        print("Applied µP initialization")
    if getattr(args, 'freeze_layers', 0) > 0:
        surgeon.freeze_layers(up_to=args.freeze_layers)
        print(f"Frozen first {args.freeze_layers} layers")

    print(f"Bridge components: {bridge.summary()}")

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        ckpt_vocab = ckpt['model_state_dict']['tok_emb.weight'].shape[0]
        cur_vocab = cfg.vocab_size
        if ckpt_vocab != cur_vocab:
            # Расширяем/обрезаем embedding до нового vocab_size
            import torch.nn as nn
            old_emb = ckpt['model_state_dict']['tok_emb.weight']   # (ckpt_vocab, d)
            new_emb = model.tok_emb.weight.data.clone()              # (cur_vocab, d)
            copy_rows = min(ckpt_vocab, cur_vocab)
            new_emb[:copy_rows] = old_emb[:copy_rows]
            ckpt['model_state_dict']['tok_emb.weight'] = new_emb
            # Аналогично для lm_head / head (разные имена в разных версиях)
            for head_key in ('lm_head.weight', 'head.weight'):
                if head_key in ckpt['model_state_dict']:
                    old_head = ckpt['model_state_dict'][head_key]
                    head_mod = model
                    for part in head_key.split('.')[:-1]:
                        head_mod = getattr(head_mod, part)
                    new_head = head_mod.weight.data.clone()
                    new_head[:copy_rows] = old_head[:copy_rows]
                    ckpt['model_state_dict'][head_key] = new_head
            print(f"Vocab extended: {ckpt_vocab} → {cur_vocab} "
                  f"({cur_vocab - ckpt_vocab:+d} новых токенов)")
        model.load_state_dict(ckpt['model_state_dict'])
        if ckpt_vocab == cur_vocab:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print("Optimizer state сброшен (vocab изменился, начинаем с нуля)")
        start_step = ckpt['step']
        print(f"Resumed from step {start_step}")

    # Данные
    data_fn = None

    if _svend4_corpus is not None:
        _svend4_corpus.print_stats()

        _use_domain_moe = getattr(cfg, 'use_domain_moe', False)

        def svend4_batch():
            if _use_domain_moe and hasattr(_svend4_corpus, 'get_batch_with_domain'):
                return _svend4_corpus.get_batch_with_domain(cfg.batch_size, device)
            return _svend4_corpus.get_batch(cfg.batch_size, device) + (None,)

        data_fn = svend4_batch
        print(f"Using svend4 corpus: {_svend4_corpus}")

    if not args.synthetic and data_fn is None:
        try:
            from data_utils.streaming_dataset import get_batch_streaming, create_train_val_iterators
            from tokenizer.tokenizer_utils import load_tokenizer
            sp = load_tokenizer()
            train_iter, _ = create_train_val_iterators()

            def streaming_batch():
                nonlocal train_iter
                xb, yb = get_batch_streaming(
                    train_iter, cfg.batch_size, cfg.block_size, device, sp
                )
                if xb is None:
                    train_iter, _ = create_train_val_iterators()
                    xb, yb = get_batch_streaming(
                        train_iter, cfg.batch_size, cfg.block_size, device, sp
                    )
                return xb, yb

            data_fn = streaming_batch
            print("Using TinyStories streaming dataset")
        except Exception as e:
            print(f"Could not load TinyStories ({e}), falling back to synthetic data")
            args.synthetic = True

    if args.synthetic:
        print("Using synthetic data with patterns")

    # Mixed precision
    use_amp = cfg.use_amp and device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32
    if use_amp:
        print("Using mixed precision (AMP) training")

    model.train()
    optimizer.zero_grad()
    accum_loss = 0.0
    start_time = time.time()

    for step in range(start_step + 1, cfg.total_steps + 1):
        # LR управляется через bridge scheduler
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        domain_ids_batch = None
        if data_fn:
            batch = data_fn()
            if len(batch) == 3:
                xb, yb, domain_ids_batch = batch
            else:
                xb, yb = batch
        else:
            xb, yb = generate_synthetic_batch(
                cfg.batch_size, cfg.block_size, cfg.vocab_size, device
            )

        # Bridge: data augmentation pipeline
        xb, yb = data_pipeline.process((xb, yb), step=step)

        # v63: обновляем Nautilus curriculum step
        if hasattr(model, 'nautilus'):
            model.nautilus.set_step(step)

        with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            logits, loss, _ = model(xb, yb, domain_ids=domain_ids_batch)
            # Bridge: модификаторы loss (Z-Loss, Entropy Reg, etc.)
            loss = bridge.before_backward(logits, yb, loss)
            loss = loss / cfg.grad_accum_steps

        scaler.scale(loss).backward()

        # Bridge: модификаторы градиентов (AGC, Grad Noise, Centralization)
        bridge.after_backward(step)

        accum_loss += loss.item()

        if step % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Bridge: мониторинг после шага (Loss Spike, Grokking, etc.)
            alerts = bridge.after_step(
                step, loss.item() * cfg.grad_accum_steps,
                lr=lr, grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            )

        if step % cfg.log_every == 0:
            avg_loss = accum_loss / cfg.log_every * cfg.grad_accum_steps
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            logger.log({
                'train_loss': avg_loss,
                'lr': lr,
                'steps_per_sec': round(steps_per_sec, 1),
            }, step)
            accum_loss = 0.0

        if step % cfg.val_every == 0:
            val_loss = estimate_val_loss(model, cfg, device, data_fn=data_fn)
            metrics = {'val_loss': val_loss}
            # Периодический анализ квантизации
            if args.model == 'yijing' and hasattr(model, 'quantization_analytics'):
                qa = model.quantization_analytics()
                for layer_name, info in qa.items():
                    for k, v in info.items():
                        if isinstance(v, (int, float)):
                            metrics[f'{layer_name}/{k}'] = v
            # v62: Convergence Bridge diagnostics (GlyphTokenizer)
            if args.model == 'yijing' and hasattr(model, 'convergence_bridge'):
                cb = model.convergence_bridge
                corr = cb.token_abstractor.cluster_hexagram_correlation().item()
                metrics['convergence/hex_correlation'] = round(corr, 4)
                metrics['convergence/bridge_scale'] = round(cb.bridge_scale.item(), 4)
                metrics['convergence/abstractor_temp'] = round(
                    cb.token_abstractor.temperature.item(), 4)
                metrics['convergence/composer_scale'] = round(
                    cb.glyph_composer.scale.item(), 4)

            # v63: Nautilus hierarchy diagnostics
            if args.model == 'yijing' and hasattr(model, 'nautilus'):
                nautilus_stats = model.nautilus.get_nautilus_stats()
                metrics.update(nautilus_stats)

            # Interlingua stats (temperature annealing, archetypes)
            if args.model == 'yijing' and hasattr(model, 'archetypal_interlingua'):
                il = model.archetypal_interlingua
                il_stats = il.get_interlingua_stats()
                metrics['interlingua/global_gate'] = il_stats['global_gate']
                metrics['interlingua/scale'] = il_stats['scale']
                trit = il_stats.get('trit_distribution', {})
                metrics['interlingua/trit_pos'] = trit.get('pos', 0)
                metrics['interlingua/trit_zero'] = trit.get('zero', 0)
                metrics['interlingua/trit_neg'] = trit.get('neg', 0)
                if 'ternary_temperature' in il_stats:
                    metrics['interlingua/ternary_temperature'] = il_stats['ternary_temperature']
                if 'active_archetypes' in il_stats:
                    metrics['interlingua/active_archetypes'] = il_stats['active_archetypes']
                    metrics['interlingua/usage_mean'] = il_stats['archetype_usage_mean']
            logger.log(metrics, step)

        if step % cfg.save_every == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(args.checkpoint_dir, f'checkpoint_step_{step}.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Финальный анализ
    if args.model == 'yijing':
        print("\n" + "=" * 50)
        print("YiJing Geometry Analysis (post-training)")
        print("=" * 50)
        contributions = measure_hex_contribution(model)
        for layer_name, vals in contributions.items():
            parts = [f"{k}={v}" if not isinstance(v, float) else f"{k}={v:.4f}"
                     for k, v in vals.items()]
            print(f"  {layer_name}: {', '.join(parts)}")

        # v62: Convergence Bridge итоговая диагностика
        if hasattr(model, 'convergence_bridge'):
            cb = model.convergence_bridge
            corr = cb.token_abstractor.cluster_hexagram_correlation().item()
            print(f"\n  ConvergenceBridge:")
            print(f"    hex_correlation={corr:.4f}")
            print(f"    bridge_scale={cb.bridge_scale.item():.4f}")
            print(f"    abstractor_temp={cb.token_abstractor.temperature.item():.4f}")
            print(f"    composer_scale={cb.glyph_composer.scale.item():.4f}")
            if model.use_glyph_tokenizer:
                print(f"    glyph_source=GlyphTokenizer (SOLAN-76)")
            else:
                print(f"    glyph_source=learned_projection")

        # v63: Nautilus итоговая диагностика
        if hasattr(model, 'nautilus'):
            print(f"\n  NautilusHierarchy (mode={model.nautilus.mode}):")
            print(f"    residual_gate={model.nautilus.residual_gate.item():.4f}")
            for name, chamber in zip(model.nautilus.chamber_names, model.nautilus.chambers):
                s = chamber.get_stats()
                print(f"    {name}: gate={s['gate_mean']:.4f}, scale={s['scale']:.4f}")

    # Bridge: финальный диагностический отчёт
    print("\n" + "=" * 50)
    print("Bridge Diagnostic Report (v12..v52 utils)")
    print("=" * 50)
    bridge_report = bridge.diagnostic_report(cfg.total_steps)
    for section, data in bridge_report.items():
        if section == 'step':
            continue
        print(f"\n  [{section}]")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
        elif isinstance(data, list):
            for item in data:
                print(f"    - {item}")
        else:
            print(f"    {data}")

    # Bridge: алерты за всю тренировку
    all_alerts = monitor.get_alerts()
    if all_alerts:
        print(f"\n  [Alerts ({len(all_alerts)} total)]")
        for alert in all_alerts[-10:]:  # последние 10
            print(f"    {alert}")

    logger.close()
    print("\nTraining complete.")
    return model


def main():
    parser = argparse.ArgumentParser(description='YiJing-Transformer Training')
    parser.add_argument('--model', type=str, default='yijing', choices=['yijing', 'vanilla'])
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--vocab-size', type=int, default=512)
    parser.add_argument('--rope', action='store_true', default=True)
    parser.add_argument('--no-rope', dest='rope', action='store_false')
    parser.add_argument('--swiglu', action='store_true', default=True)
    parser.add_argument('--no-swiglu', dest='swiglu', action='store_false')
    parser.add_argument('--bian-gua', action='store_true', default=True)
    parser.add_argument('--no-bian-gua', dest='bian_gua', action='store_false')
    parser.add_argument('--moe', action='store_true', default=False)
    parser.add_argument('--domain-moe', action='store_true', default=False,
                        help='DomainMoE: эксперты специализируются по доменам корпуса')
    parser.add_argument('--domain-moe-experts', type=int, default=6,
                        help='Число экспертов DomainMoE (по умолчанию 6 = число доменов)')
    parser.add_argument('--domain-moe-top-k', type=int, default=2,
                        help='Число активных экспертов за forward')
    parser.add_argument('--domain-supervision-weight', type=float, default=0.1,
                        help='Вес loss доменной специализации')
    parser.add_argument('--adaptive-temp', action='store_true', default=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--synthetic', action='store_true', default=False)
    parser.add_argument('--svend4', type=str, default=None,
                        metavar='CORPUS_DIR',
                        help='Путь к корпусу svend4 (data/svend4_corpus). '
                             'Запустите scripts/fetch_svend4_corpus.py для загрузки.')
    parser.add_argument('--svend4-domains', type=str, default=None,
                        help='Домены через запятую: ai_agents,infosystems,knowledge,algorithms')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--tb-dir', type=str, default='runs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Mixed precision training (requires CUDA)')
    parser.add_argument('--gqa-heads', type=int, default=None,
                        help='Number of KV heads for GQA (default: MHA)')
    parser.add_argument('--sliding-window', type=int, default=None,
                        help='Sliding window size for attention')
    parser.add_argument('--glyph-tokenizer', action='store_true', default=False,
                        help='Use GlyphTokenizer (SOLAN-76) for ConvergenceBridge Q6 vertices')
    parser.add_argument('--convergence-bridge', action='store_true', default=False,
                        help='Enable ConvergenceBridge (without GlyphTokenizer uses learned projection)')
    parser.add_argument('--nautilus', action='store_true', default=False,
                        help='Enable NautilusHierarchy — hierarchical geometric modules')
    parser.add_argument('--nautilus-mode', type=str, default='sequential',
                        choices=['sequential', 'parallel'],
                        help='Nautilus mode: sequential (cascade) or parallel (merge)')
    parser.add_argument('--nautilus-chambers', type=str, default='all',
                        help='Comma-separated chamber names or "all"')
    parser.add_argument('--nautilus-init-scale', type=float, default=0.01,
                        help='Initial scale for nautilus chambers')
    parser.add_argument('--nautilus-warmup-steps', type=int, default=2000,
                        help='Steps for progressive chamber activation')

    # === Bridge: утилиты из v12..v52 ===
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'sophia', 'lamb', 'lion'],
                        help='Optimizer type (v18: LAMB, v19: Sophia, v45: Lion)')
    parser.add_argument('--optimizer-wrapper', type=str, default=None,
                        choices=['lookahead', 'sam'],
                        help='Optimizer wrapper (v23: Lookahead, v27: SAM)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'wsd', 'cosine_restarts'],
                        help='LR scheduler (v18: WSD, v32: Cosine Restarts)')
    parser.add_argument('--llrd-factor', type=float, default=1.0,
                        help='Layer-wise LR decay factor (v14/v24: 0.8 typical)')
    parser.add_argument('--mup-init', action='store_true', default=False,
                        help='Apply µP initialization (v12)')
    parser.add_argument('--freeze-layers', type=int, default=0,
                        help='Freeze first N layers (v50: progressive freezing)')
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

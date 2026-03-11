#!/usr/bin/env python3
"""
v61 Diagnostic: Почему архетипы НЕ активируются?

Запускает ОДНУ модель (bridged_high_uncertainty) с подробным логированием:
1. raw trit scores (до квантизации) — насколько далеко от порога?
2. gradient norms по ключевым компонентам
3. archetype queries evolution
4. threshold vs max|raw| — видно ли движение к порогу?
5. contribution norms — что bridge-encoders выдают?

Гипотеза: STE zero-gradient trap — когда все hard=0, weights=0,
градиент не течёт через mean_contrib * weights, и система застревает.

Использование:
  python benchmark_v61_diagnostic.py
"""

import sys
import os
import math
import time
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from yijing_transformer.config import YiJingConfig
from yijing_transformer.models.model import YiJingGPT

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'benchmark_v61_diagnostic_results.json'
)


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def load_data(block_size=128, seed=42):
    random.seed(seed)
    words = [
        "the", "of", "and", "to", "in", "a", "is", "that", "it", "was",
        "for", "on", "are", "with", "as", "his", "they", "be", "at", "one",
        "have", "this", "from", "by", "not", "but", "what", "all", "were", "when",
        "we", "there", "can", "an", "your", "which", "their", "said", "each", "she",
        "do", "how", "will", "up", "other", "about", "out", "many", "then", "them",
        "would", "like", "so", "these", "her", "long", "make", "thing", "see", "him",
        "two", "has", "look", "more", "day", "could", "go", "come", "did", "my",
        "no", "most", "who", "over", "know", "than", "call", "first", "people", "may",
        "down", "been", "now", "find", "any", "new", "work", "part", "take", "get",
        "place", "made", "after", "back", "only", "use", "where", "good", "very", "still",
    ]
    lines = []
    for _ in range(60000):
        n = random.randint(5, 25)
        line = ' '.join(random.choice(words) for _ in range(n))
        lines.append(line + '.')
    full = '\n'.join(lines)
    split = int(len(full) * 0.9)
    train_text, val_text = full[:split], full[split:]
    train_data = torch.tensor(list(train_text.encode('utf-8')), dtype=torch.long)
    val_data = torch.tensor(list(val_text.encode('utf-8')), dtype=torch.long)
    print(f"  Train: {len(train_data):,}, Val: {len(val_data):,} tokens (byte-level)")
    return train_data, val_data, 256


def get_batch(data, block_size, batch_size):
    n = len(data) - block_size - 1
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


ALL_SOURCES = dict(
    hex_strength=0.05,
    quantizer_type='factored6',
    use_heisenberg_attention=True,
    use_flower_gat=True,
    use_palace_attention=True,
    use_privileged_axis=True,
    use_cube_diagonal=True,
    use_dual_embedding=True,
    use_d4_equivariant=True,
)


def make_config(overrides, vocab_size, d_model=128, n_layers=4, n_heads=4, block_size=128):
    base = dict(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, block_size=block_size, dropout=0.05,
        use_rope=True, use_swiglu=True, temp=0.3,
    )
    return YiJingConfig(**{**base, **overrides})


@torch.no_grad()
def evaluate(model, val_data, block_size, batch_size, n_eval=30):
    model.eval()
    losses = []
    for _ in range(n_eval):
        x, y = get_batch(val_data, block_size, batch_size)
        logits, loss, _ = model(x, targets=y)
        if isinstance(loss, torch.Tensor):
            losses.append(loss.item())
        else:
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            ).item())
    avg = sum(losses) / len(losses)
    return avg, math.exp(min(avg, 20))


def get_interlingua_module(model):
    """Найти BridgedInterlingua модуль."""
    for layer in model.core.layers:
        if hasattr(layer, 'archetypal_interlingua'):
            return layer.archetypal_interlingua
    return None


def diagnose_trit_scores(il_module):
    """Собрать диагностику raw trit scores (требует hook-а на forward)."""
    stats = {}

    if not il_module.use_ternary:
        return stats

    threshold = (1.0 - il_module.uncertainty_budget) * 0.5 + 0.1
    stats['threshold'] = threshold.item()

    if il_module._last_archetype_usage is not None:
        usage = il_module._last_archetype_usage
        stats['usage_mean'] = usage.mean().item()
        stats['usage_max'] = usage.max().item()
        stats['usage_min'] = usage.min().item()
        stats['active_count'] = (usage > 0.1).sum().item()

    # Проверим параметры trit_proj
    w = il_module.trit_proj.weight
    b = il_module.trit_proj.bias
    stats['trit_proj_weight_norm'] = w.norm().item()
    stats['trit_proj_weight_max'] = w.abs().max().item()
    stats['trit_proj_bias'] = b.item()

    # Archetype queries stats
    aq = il_module.archetype_queries
    stats['archetype_queries_norm'] = aq.norm(dim=1).mean().item()
    stats['archetype_queries_max'] = aq.abs().max().item()

    # Scale и gate
    stats['global_gate'] = torch.sigmoid(il_module.global_gate).item()
    stats['scale'] = il_module.scale.item()

    # Bridge scales
    stats['bridge_scales'] = [b.scale.item() for b in il_module.bridges]

    # log_uncertainty
    stats['log_uncertainty'] = il_module.log_uncertainty.item()
    stats['uncertainty_budget'] = il_module.uncertainty_budget.item()

    return stats


def diagnose_gradients(model):
    """Собрать gradient norms по ключевым группам параметров."""
    il = get_interlingua_module(model)
    if il is None:
        return {}

    grad_stats = {}

    def _grad_norm(name, params):
        total = 0.0
        count = 0
        for p in params:
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
                count += 1
        if count > 0:
            grad_stats[name] = math.sqrt(total)
            grad_stats[f'{name}_count'] = count
        else:
            grad_stats[name] = 0.0
            grad_stats[f'{name}_count'] = 0

    _grad_norm('trit_proj', il.trit_proj.parameters())
    _grad_norm('archetype_queries', [il.archetype_queries])
    _grad_norm('bridge_encoders', il.bridge_encoders.parameters())
    _grad_norm('encode_attn', il.encode_attn.parameters())
    _grad_norm('readout_attn', il.readout_attn.parameters())
    _grad_norm('readout_proj', il.readout_proj.parameters())
    _grad_norm('aggregate_proj', il.aggregate_proj.parameters())
    _grad_norm('global_gate', [il.global_gate])
    _grad_norm('scale_param', [il.scale])
    _grad_norm('log_uncertainty', [il.log_uncertainty])

    return grad_stats


class TritScoreHook:
    """Hook для захвата raw scores из _ternary_quantize."""

    def __init__(self):
        self.last_raw_scores = []  # list of (B, n_archetypes) tensors
        self.last_hard_trits = []
        self.enabled = True

    def install(self, il_module):
        """Monkey-patch _ternary_quantize для захвата raw scores."""
        original = il_module._ternary_quantize

        hook = self

        def patched_quantize(contribution):
            scores = il_module.trit_proj(contribution).squeeze(-1)
            raw = torch.tanh(scores)

            threshold = (1.0 - il_module.uncertainty_budget) * 0.5 + 0.1

            hard = torch.zeros_like(raw)
            hard[raw > threshold] = 1.0
            hard[raw < -threshold] = -1.0
            trit_scores = raw + (hard - raw).detach()

            if hook.enabled:
                hook.last_raw_scores.append(raw.detach().clone())
                hook.last_hard_trits.append(hard.detach().clone())

            return trit_scores

        il_module._ternary_quantize = patched_quantize
        return self

    def get_stats(self):
        if not self.last_raw_scores:
            return {}
        # Stack all bridge outputs: (K, B, n_archetypes)
        raw = torch.stack(self.last_raw_scores, dim=0)
        hard = torch.stack(self.last_hard_trits, dim=0)

        stats = {
            'raw_mean': raw.mean().item(),
            'raw_std': raw.std().item(),
            'raw_abs_mean': raw.abs().mean().item(),
            'raw_abs_max': raw.abs().max().item(),
            'raw_abs_min': raw.abs().min().item(),
            # Per-archetype: максимальный |raw| по всем bridge-выходам и батчам
            'raw_abs_max_per_archetype_mean': raw.abs().amax(dim=(0, 1)).mean().item(),
            'raw_abs_max_per_archetype_max': raw.abs().amax(dim=(0, 1)).max().item(),
            # Сколько раз raw превысил порог
            'n_positive': (hard > 0).sum().item(),
            'n_negative': (hard < 0).sum().item(),
            'n_zero': (hard == 0).sum().item(),
            'total_trits': hard.numel(),
        }
        return stats

    def clear(self):
        self.last_raw_scores.clear()
        self.last_hard_trits.clear()


def run_diagnostic():
    print("=" * 70)
    print("  v61 DIAGNOSTIC: Почему архетипы НЕ активируются?")
    print("  Модель: bridged_high_uncertainty (uncertainty=0.7, threshold≈0.25)")
    print("=" * 70)

    print("\n  Loading data...")
    train_data, val_data, vocab_size = load_data()

    config_overrides = dict(
        **ALL_SOURCES,
        use_bridged_interlingua=True,
        bridged_bridge_mode='lightweight',
        interlingua_use_ternary=True,
        interlingua_uncertainty=0.7,
        interlingua_n_archetypes=64,
        interlingua_n_heads=4,
    )

    set_seed(42)
    cfg = make_config(config_overrides, vocab_size)
    model = YiJingGPT(cfg)

    il = get_interlingua_module(model)
    assert il is not None, "BridgedInterlingua не найден!"

    trit_hook = TritScoreHook().install(il)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} params")
    print(f"  BridgedInterlingua: n_archetypes={il.n_archetypes}, "
          f"n_bridges={il.n_pairs}, n_bridge_outputs={il.n_bridge_outputs}")
    print(f"  Initial threshold: {((1.0 - il.uncertainty_budget) * 0.5 + 0.1).item():.4f}")
    print(f"  Initial uncertainty: {il.uncertainty_budget.item():.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    n_steps = 800
    batch_size = 16
    lr = 1e-3
    eval_every = 50  # чаще для диагностики (было 200)
    diag_every = 25  # ещё чаще для raw scores

    history = []
    diag_history = []
    grad_history = []
    t0 = time.time()
    best_val = float('inf')

    print(f"\n  Training {n_steps} steps, eval every {eval_every}, diag every {diag_every}")
    print(f"  {'Step':>6} {'Train':>8} {'Val':>8} {'PPL':>6} {'|raw|max':>9} "
          f"{'|raw|mean':>9} {'threshold':>9} {'active':>7} {'gate':>6} "
          f"{'∇trit':>8} {'∇enc':>8} {'∇readout':>8}")
    print("  " + "-" * 110)

    for step in range(1, n_steps + 1):
        model.train()
        progress = step / n_steps
        cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        x, y = get_batch(train_data, cfg.block_size, batch_size)

        trit_hook.clear()

        logits, loss, _ = model(x, targets=y)
        if not isinstance(loss, torch.Tensor):
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # --- Диагностика ПЕРЕД optimizer.step() ---
        do_diag = (step % diag_every == 0 or step == 1 or step <= 5)
        do_eval = (step % eval_every == 0 or step == 1)

        if do_diag:
            trit_stats = trit_hook.get_stats()
            il_diag = diagnose_trit_scores(il)
            grad_stats = diagnose_gradients(model)

            raw_abs_max = trit_stats.get('raw_abs_max', 0)
            raw_abs_mean = trit_stats.get('raw_abs_mean', 0)
            threshold = il_diag.get('threshold', 0)
            active = il_diag.get('active_count', 0)
            gate = il_diag.get('global_gate', 0)
            grad_trit = grad_stats.get('trit_proj', 0)
            grad_enc = grad_stats.get('bridge_encoders', 0)
            grad_readout = grad_stats.get('readout_attn', 0)

            diag_entry = {
                'step': step,
                'train_loss': loss.item(),
                'trit_scores': trit_stats,
                'il_diag': il_diag,
                'grad_stats': grad_stats,
            }
            diag_history.append(diag_entry)

            if do_eval:
                vl, ppl = evaluate(model, val_data, cfg.block_size, batch_size)
                best_val = min(best_val, vl)
                history.append({'step': step, 'train': loss.item(), 'val': vl, 'ppl': ppl})
                print(f"  {step:6d} {loss.item():8.4f} {vl:8.4f} {ppl:6.1f} "
                      f"{raw_abs_max:9.5f} {raw_abs_mean:9.5f} {threshold:9.5f} "
                      f"{active:7d} {gate:6.3f} "
                      f"{grad_trit:8.5f} {grad_enc:8.5f} {grad_readout:8.5f}")
            else:
                print(f"  {step:6d} {loss.item():8.4f} {'':>8} {'':>6} "
                      f"{raw_abs_max:9.5f} {raw_abs_mean:9.5f} {threshold:9.5f} "
                      f"{active:7d} {gate:6.3f} "
                      f"{grad_trit:8.5f} {grad_enc:8.5f} {grad_readout:8.5f}")
        elif do_eval:
            vl, ppl = evaluate(model, val_data, cfg.block_size, batch_size)
            best_val = min(best_val, vl)
            history.append({'step': step, 'train': loss.item(), 'val': vl, 'ppl': ppl})
            print(f"  {step:6d} {loss.item():8.4f} {vl:8.4f} {ppl:6.1f}")

        optimizer.step()

    elapsed = time.time() - t0

    # --- Финальная диагностика ---
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")

    final_il = diagnose_trit_scores(il)
    final_trit = trit_hook.get_stats()

    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Best val loss: {best_val:.4f} (PPL {math.exp(min(best_val, 20)):.1f})")

    print(f"\n  --- Trit Score Analysis ---")
    print(f"  Final threshold: {final_il.get('threshold', '?'):.4f}")
    print(f"  Final |raw| max: {final_trit.get('raw_abs_max', '?')}")
    print(f"  Final |raw| mean: {final_trit.get('raw_abs_mean', '?')}")
    print(f"  Positive trits: {final_trit.get('n_positive', 0)} / {final_trit.get('total_trits', 0)}")
    print(f"  Negative trits: {final_trit.get('n_negative', 0)} / {final_trit.get('total_trits', 0)}")
    print(f"  Zero trits: {final_trit.get('n_zero', 0)} / {final_trit.get('total_trits', 0)}")

    print(f"\n  --- Key Parameters ---")
    print(f"  trit_proj weight norm: {final_il.get('trit_proj_weight_norm', '?')}")
    print(f"  trit_proj bias: {final_il.get('trit_proj_bias', '?')}")
    print(f"  archetype queries norm (mean): {final_il.get('archetype_queries_norm', '?')}")
    print(f"  global_gate (sigmoid): {final_il.get('global_gate', '?')}")
    print(f"  scale: {final_il.get('scale', '?')}")
    print(f"  uncertainty_budget: {final_il.get('uncertainty_budget', '?')}")

    print(f"\n  --- Evolution Over Training ---")
    if diag_history:
        key_steps = [diag_history[0], diag_history[len(diag_history)//4],
                     diag_history[len(diag_history)//2],
                     diag_history[3*len(diag_history)//4],
                     diag_history[-1]]
        print(f"  {'Step':>6} {'|raw|max':>10} {'|raw|mean':>10} {'threshold':>10} "
              f"{'∇trit':>10} {'∇enc':>10} {'gate':>8} {'scale':>8}")
        for d in key_steps:
            s = d['step']
            ts = d['trit_scores']
            gs = d['grad_stats']
            il_d = d['il_diag']
            print(f"  {s:6d} {ts.get('raw_abs_max', 0):10.6f} "
                  f"{ts.get('raw_abs_mean', 0):10.6f} "
                  f"{il_d.get('threshold', 0):10.6f} "
                  f"{gs.get('trit_proj', 0):10.6f} "
                  f"{gs.get('bridge_encoders', 0):10.6f} "
                  f"{il_d.get('global_gate', 0):8.4f} "
                  f"{il_d.get('scale', 0):8.4f}")

    # Определяем корневую причину
    print(f"\n  --- ROOT CAUSE ANALYSIS ---")

    if diag_history:
        final_d = diag_history[-1]
        raw_max = final_d['trit_scores'].get('raw_abs_max', 0)
        thresh = final_d['il_diag'].get('threshold', 1.0)
        grad_trit = final_d['grad_stats'].get('trit_proj', 0)
        grad_enc = final_d['grad_stats'].get('bridge_encoders', 0)

        if raw_max < thresh * 0.5:
            print(f"  [!] raw scores far below threshold ({raw_max:.4f} << {thresh:.4f})")
            print(f"      Scores never approach activation boundary")

        if grad_trit < 1e-6:
            print(f"  [!] ZERO gradient on trit_proj ({grad_trit:.8f})")
            print(f"      Confirms: STE zero-gradient trap!")
            print(f"      When all hard=0, weights=|consensus|=0, and")
            print(f"      d(loss)/d(mean_contrib) = 0 because mean_contrib * 0 = 0")
        elif grad_trit < 1e-3:
            print(f"  [!] Very small gradient on trit_proj ({grad_trit:.6f})")
            print(f"      Gradient exists but too weak to push past threshold")

        if grad_enc < 1e-6:
            print(f"  [!] ZERO gradient on bridge_encoders ({grad_enc:.8f})")
            print(f"      Encoders are not learning at all")

        first_raw_max = diag_history[0]['trit_scores'].get('raw_abs_max', 0)
        if abs(raw_max - first_raw_max) < 0.01:
            print(f"  [!] raw scores did NOT move during training")
            print(f"      Step 1 max: {first_raw_max:.6f}, Final max: {raw_max:.6f}")

    # Сохраняем результаты
    results = {
        'config': 'bridged_high_uncertainty',
        'n_steps': n_steps,
        'n_params': n_params,
        'best_val': best_val,
        'best_ppl': math.exp(min(best_val, 20)),
        'time': elapsed,
        'history': history,
        'diagnostic_history': diag_history,
        'final_trit_stats': final_trit,
        'final_il_diag': final_il,
        '_last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == '__main__':
    run_diagnostic()

#!/usr/bin/env python3
"""
train_all_models.py — Обучение всех моделей на info_corpus

Обучает 7 моделей индивидуально, затем Grand Orchestrator как ансамбль.
"""

import sys, os, time, random, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'yijing_transformer', 'scripts'))

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════
# Токенизатор
# ═══════════════════════════════════════════════════════════════

def load_tokenizer():
    try:
        import sentencepiece as spm
        path = os.path.join(ROOT, 'yijing_transformer', 'bpe_tokenizer.model')
        sp = spm.SentencePieceProcessor()
        sp.load(path)
        return sp
    except Exception:
        return None

class Tokenizer:
    def __init__(self, sp=None):
        self.sp = sp
    def encode(self, text):
        if self.sp:
            return self.sp.encode(text)
        return [min(ord(ch) + 2, 4095) for ch in text]
    def decode(self, ids):
        if self.sp:
            return self.sp.decode(ids)
        return ''.join(chr(max(i - 2, 0)) for i in ids if i > 1)


# ═══════════════════════════════════════════════════════════════
# Загрузка корпуса
# ═══════════════════════════════════════════════════════════════

def load_corpus(tokenizer, block_size=64):
    """Загрузка и токенизация info_corpus."""
    corpus_path = os.path.join(ROOT, 'data', 'info_corpus', 'combined_corpus.txt')
    if not os.path.exists(corpus_path):
        # Попробуем собрать из отдельных файлов
        md_files = sorted(glob.glob(os.path.join(ROOT, 'data', 'info_corpus', '*.md')))
        if not md_files:
            print("  ОШИБКА: корпус не найден!")
            return []
        texts = []
        for f in md_files:
            with open(f, 'r', encoding='utf-8') as fh:
                texts.append(fh.read())
        full_text = '\n\n'.join(texts)
    else:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

    print(f"  Корпус: {len(full_text):,} символов")

    # Разбиваем на параграфы
    paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 20]
    print(f"  Параграфов: {len(paragraphs)}")

    # Токенизируем
    all_sequences = []
    for p in paragraphs:
        ids = tokenizer.encode(p)
        if len(ids) >= 10:
            # Разрезаем длинные на block_size+1
            for i in range(0, len(ids) - block_size, block_size // 2):
                seq = ids[i:i + block_size + 1]
                if len(seq) == block_size + 1:
                    all_sequences.append(seq)
            # Последний кусок с паддингом
            if len(ids) > block_size + 1:
                seq = ids[-block_size - 1:]
                all_sequences.append(seq)
            elif len(ids) <= block_size + 1:
                padded = ids + [0] * (block_size + 1 - len(ids))
                all_sequences.append(padded[:block_size + 1])

    random.shuffle(all_sequences)
    print(f"  Обучающих последовательностей: {len(all_sequences)}")
    return all_sequences


def get_batch(sequences, batch_size=8, block_size=64):
    """Получить батч из данных."""
    idxs = [random.randint(0, len(sequences) - 1) for _ in range(batch_size)]
    xs, ys = [], []
    for i in idxs:
        seq = sequences[i]
        xs.append(seq[:block_size])
        ys.append(seq[1:block_size + 1])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════
# Создание моделей
# ═══════════════════════════════════════════════════════════════

def create_all_models():
    """Создаём все 7 моделей."""
    models = {}

    # 1. NautilusMoME (загружаем обученную)
    try:
        from train_nautilus_mome import NautilusMoME
        ckpt_path = os.path.join(ROOT, 'yijing_transformer', 'train_mome_checkpoint.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            args = ckpt.get('args', {})
            model = NautilusMoME(
                vocab_size=args.get('vocab_size', 4096),
                d_model=args.get('d_model', 128),
                n_layers=args.get('n_layers', 4),
                n_heads=args.get('n_heads', 4),
                block_size=args.get('block_size', 256),
                d_expert=args.get('d_expert', 128),
                n_experts=args.get('n_experts', 6),
                top_k=args.get('top_k', 2),
            )
            model.load_state_dict(ckpt['model'], strict=False)
            models['nautilus_mome'] = model
            print(f"  ✓ nautilus_mome — из чекпоинта ({ckpt.get('step', '?')} шагов)")
        else:
            print(f"  ✗ nautilus_mome — чекпоинт не найден")
    except Exception as e:
        print(f"  ✗ nautilus_mome — {e}")

    # 2. PolyglotQuartet (новая)
    try:
        from yijing_transformer.models.polyglot import build_polyglot
        models['quartet'] = build_polyglot(4096, 128, 2, block_size=256)
        print(f"  ✓ quartet — новая")
    except Exception as e:
        print(f"  ✗ quartet — {e}")

    # 3. Variant3GPT (новая)
    try:
        from yijing_transformer.models.variant3 import Variant3GPT, Variant3Config
        cfg = Variant3Config(vocab_size=4096, d_model=128, n_layers=2, n_heads=4, block_size=256)
        models['variant3'] = Variant3GPT(cfg)
        print(f"  ✓ variant3 — новая")
    except Exception as e:
        print(f"  ✗ variant3 — {e}")

    # 4. YiJingGPT (из чекпоинта с расширением vocab)
    try:
        from yijing_transformer.models.model import YiJingGPT
        ckpt_path = os.path.join(ROOT, 'yijing_transformer', 'train_real_data_checkpoint.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            sd = ckpt.get('model', ckpt)
            v, d = sd.get('tok_emb.weight', torch.zeros(256, 128)).shape if 'tok_emb.weight' in sd else (256, 128)

            class Cfg:
                pass
            cfg = Cfg()
            cfg.vocab_size = v; cfg.d_model = d; cfg.block_size = 256
            cfg.n_layers = 4; cfg.n_heads = max(d // 64, 2); cfg.ffn_mult = 4
            cfg.dropout = 0.05; cfg.use_rope = True; cfg.weight_tying = True
            cfg.label_smoothing = 0.0
            for attr in [
                'use_four_level_pe', 'use_cubic_pe', 'use_bidirectional_tri',
                'use_convergence_bridge', 'use_matrix_grammar', 'use_abriale',
                'use_nautilus', 'use_pseudo_rag', 'use_diff_attn',
                'use_expert_choice', 'use_six_sources', 'use_alibi',
                'use_glyph_tokenizer', 'use_glyph_prior', 'use_gradient_ckpt',
                'use_gumbel', 'use_hex_moe', 'use_swiglu', 'use_flash_attn',
                'use_bian_gua', 'use_quadrant_attention',
            ]:
                setattr(cfg, attr, False)
            cfg.bias = False; cfg.n_kv_heads = None; cfg.sliding_window = None
            cfg.attention_sinks = 0; cfg.rope_base = 10000; cfg.rope_scaling = None
            cfg.rope_scaling_factor = 1.0; cfg.quantizer_type = 'factored6'
            cfg.quant_total_dim = 6; cfg.quant_dim_schedule = None
            cfg.quant_group_dim = 6; cfg.multi_scale_quant = False
            cfg.temp = 1.0; cfg.adaptive_temp = False; cfg.commitment_weight = 0.25
            cfg.head_dim = d // cfg.n_heads; cfg.n_experts = 0; cfg.moe_top_k = 2
            cfg.hex_strength = 0.1; cfg.gate_init_bias = 0.0; cfg.total_steps = 10000
            cfg.token_merge_ratio = 0.0; cfg.prefix_len = 0; cfg.mtp_n_future = 0
            cfg.ffn_hidden = d * cfg.ffn_mult; cfg.distill_temp = 2.0
            cfg.pseudo_rag_distill_weight = 0.1
            cfg.curriculum_strategy_geo = 'linear'; cfg.curriculum_target_strength = 1.0
            cfg.curriculum_warmup_fraction = 0.1
            cfg.convergence_n_clusters = 64; cfg.convergence_window_size = 4
            cfg.convergence_stride = 2; cfg.convergence_compose_layers = 1
            cfg.convergence_n_heads = 4
            cfg.abriale_d_event = 64; cfg.abriale_n_heads = 4; cfg.abriale_arity = 2
            cfg.abriale_n_rules = 64; cfg.abriale_n_hits = 4
            cfg.abriale_n_alternatives = 2; cfg.abriale_n_event_types = 8
            cfg.abriale_balance_weight = 0.01
            cfg.nautilus_chambers = 'all'; cfg.nautilus_init_scale = 0.01
            cfg.nautilus_warmup_steps = 2000; cfg.nautilus_mode = 'sequential'
            cfg.matrix_grammar_rows = 8; cfg.matrix_grammar_cols = 8
            cfg.matrix_grammar_heads = 4

            model = YiJingGPT(cfg)
            model.load_state_dict(sd, strict=False)

            # Расширяем vocab 256 → 4096
            target_vocab = 4096
            if v < target_vocab:
                old_emb = model.tok_emb.weight.data
                new_emb = nn.Embedding(target_vocab, d)
                nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
                new_emb.weight.data[:v] = old_emb
                model.tok_emb = new_emb
                if hasattr(model, 'head') and model.head is not None:
                    old_head = model.head
                    new_head = nn.Linear(d, target_vocab, bias=old_head.bias is not None)
                    nn.init.normal_(new_head.weight, mean=0.0, std=0.02)
                    if new_head.bias is not None:
                        nn.init.zeros_(new_head.bias)
                    new_head.weight.data[:v] = old_head.weight.data
                    if old_head.bias is not None and new_head.bias is not None:
                        new_head.bias.data[:v] = old_head.bias.data
                    model.head = new_head
                cfg.vocab_size = target_vocab

            models['yijing'] = model
            print(f"  ✓ yijing — из чекпоинта (vocab {v}→4096)")
        else:
            print(f"  ✗ yijing — чекпоинт не найден")
    except Exception as e:
        print(f"  ✗ yijing — {e}")

    # 5. HierarchicalE2
    try:
        from yijing_transformer.models.hierarchical_e2 import HierarchicalE2, E2Config
        cfg = E2Config(vocab_size=4096, d_model=128, block_size=256, n_core=2, n_heads=4)
        models['hierarchical_e2'] = HierarchicalE2(cfg)
        print(f"  ✓ hierarchical_e2 — новая")
    except Exception as e:
        print(f"  ✗ hierarchical_e2 — {e}")

    # 6. NautilusYiJing
    try:
        from yijing_transformer.models.nautilus_yijing import NautilusYiJing, NautilusYiJingConfig
        cfg = NautilusYiJingConfig(
            vocab_size=4096, d_model=128, block_size=256,
            n_layers=4, n_heads=4, d_expert=64, n_experts=6, top_k=2,
            dropout=0.05, enable_synth=False,
        )
        models['nautilus_yijing'] = NautilusYiJing(cfg)
        print(f"  ✓ nautilus_yijing — новая")
    except Exception as e:
        print(f"  ✗ nautilus_yijing — {e}")

    # 7. HierarchicalMoE (из чекпоинта)
    try:
        from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
        from yijing_transformer.models.hierarchical_moe import HMoEConfig, HierarchicalMoEFFN
        from yijing_transformer.models.geometry.routing import ArchetypalInterlingua

        ckpt_path = os.path.join(ROOT, 'hmoe_fixed_joint.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            sd = ckpt.get('model_state', ckpt)
            model_cfg = dict(
                vocab_size=256, block_size=64, d_model=128,
                n_heads=4, n_layers=4, ffn_mult=4,
                hamming_lambda=0.15, uncertainty_budget=0.25,
                dropout=0.1, use_domain_routing=False,
                use_hierarchical_moe=True,
            )
            hmoe_cfg = HMoEConfig(d_model=128, use_multiscale=True, use_hex_tier=False)
            cfg = Variant3Config(**model_cfg)
            model = Variant3GPT(cfg)
            for block in model.blocks:
                if hasattr(block, 'hmoe'):
                    block.hmoe = HierarchicalMoEFFN(hmoe_cfg)

            def _make_adapted(il):
                _orig = il.forward
                def _f(source_outputs, core_hidden):
                    out = _orig(core_hidden, source_outputs)
                    aux = il.get_interlingua_loss() if hasattr(il, 'get_interlingua_loss') else 0.0
                    return out, aux
                return _f

            for block in model.blocks:
                if hasattr(block, 'interlingua'):
                    il = ArchetypalInterlingua(d_model=128, n_sources=2, n_archetypes=64,
                                               uncertainty_budget=0.25)
                    il.forward = _make_adapted(il)
                    block.interlingua = il

            model.load_state_dict(sd, strict=False)
            models['hmoe'] = model
            print(f"  ✓ hmoe — из чекпоинта")
        else:
            print(f"  ✗ hmoe — чекпоинт не найден")
    except Exception as e:
        print(f"  ✗ hmoe — {e}")

    return models


# ═══════════════════════════════════════════════════════════════
# Обучение одной модели
# ═══════════════════════════════════════════════════════════════

def train_model(name, model, sequences, steps=500, lr=3e-4, batch_size=8, block_size=64):
    """Обучение одной модели."""
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"\n{'─' * 60}")
    print(f"  Обучение: {name} ({n_params:,} обучаемых параметров)")
    print(f"  Шагов: {steps}, lr={lr}, batch={batch_size}")
    print(f"{'─' * 60}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.1)

    t0 = time.time()
    losses = []
    best_loss = float('inf')

    for step in range(steps):
        x, y = get_batch(sequences, batch_size, block_size)

        # Обработка для моделей с маленьким vocab (hmoe: vocab=256)
        model_vocab = None
        if hasattr(model, 'tok_emb'):
            model_vocab = model.tok_emb.weight.shape[0]
        elif hasattr(model, 'embedding'):
            model_vocab = model.embedding.weight.shape[0]

        if model_vocab and model_vocab < 4096:
            x = x.clamp(0, model_vocab - 1)
            y = y.clamp(0, model_vocab - 1)

        try:
            result = model(x, targets=y)
        except TypeError:
            try:
                result = model(x, y)
            except TypeError:
                result = model(x)

        if isinstance(result, tuple):
            if len(result) >= 2:
                logits, loss = result[0], result[1]
            else:
                logits = result[0]
                loss = None
        else:
            logits = result
            loss = None

        if loss is None or (hasattr(loss, 'item') and loss.item() == 0):
            # Вычисляем loss вручную
            if logits.dim() == 3:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=0,
                )

        if loss is None:
            print(f"  ОШИБКА: не удалось вычислить loss для {name}")
            return losses

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        l = loss.item()
        losses.append(l)
        if l < best_loss:
            best_loss = l

        if step % 100 == 0 or step == steps - 1:
            avg = sum(losses[-100:]) / len(losses[-100:])
            elapsed = time.time() - t0
            speed = (step + 1) / elapsed
            print(f"  шаг {step:4d}/{steps} | loss={avg:.4f} | best={best_loss:.4f} | {speed:.1f} шаг/с")

    elapsed = time.time() - t0
    final_avg = sum(losses[-50:]) / len(losses[-50:])
    print(f"  Итого: {elapsed:.1f}с, loss {losses[0]:.3f} → {final_avg:.4f} "
          f"(снижение {(1 - final_avg / losses[0]) * 100:.1f}%)")

    return losses


# ═══════════════════════════════════════════════════════════════
# Генерация текста
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(model, tokenizer, prompt, max_len=80, temperature=0.8, block_size=64):
    """Генерация текста одной моделью."""
    model.eval()
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)

    # Определяем vocab модели
    model_vocab = None
    if hasattr(model, 'tok_emb'):
        model_vocab = model.tok_emb.weight.shape[0]
    elif hasattr(model, 'embedding'):
        model_vocab = model.embedding.weight.shape[0]

    for _ in range(max_len):
        idx_cond = idx[:, -block_size:]
        if model_vocab and model_vocab < 4096:
            idx_cond = idx_cond.clamp(0, model_vocab - 1)

        try:
            result = model(idx_cond)
        except Exception:
            break

        if isinstance(result, tuple):
            logits = result[0]
        else:
            logits = result

        logits = logits[:, -1, :] / temperature

        # top-k sampling
        top_k = 40
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        # Стоп на знаке препинания
        ch = tokenizer.decode([next_id.item()])
        if ch and ch[-1] in '.!?\n':
            break

    return tokenizer.decode(idx[0].tolist())


# ═══════════════════════════════════════════════════════════════
# Обучение оркестратора
# ═══════════════════════════════════════════════════════════════

def train_orchestrator_ensemble(models, sequences, tokenizer, steps=300, lr=2e-3, batch_size=8, block_size=64):
    """Обучение Grand Orchestrator."""
    from yijing_transformer.models.grand_orchestrator import build_grand_orchestrator

    print(f"\n{'═' * 60}")
    print(f"  ОБУЧЕНИЕ GRAND ORCHESTRATOR")
    print(f"{'═' * 60}")

    best_mode = None
    best_loss = float('inf')
    best_orch = None

    for mode in ['blend', 'expert', 'cascade']:
        print(f"\n  --- Режим: {mode.upper()} ---")
        orch = build_grand_orchestrator(
            models, mode=mode, vocab_size=4096, d_model=128,
            freeze=False, expert_top_k=3,
        )
        orch.train()
        trainable = [p for p in orch.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

        t0 = time.time()
        losses = []

        for step in range(steps):
            x, y = get_batch(sequences, batch_size, block_size)
            logits, loss, info = orch(x, targets=y, mode=mode)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(loss.item())

            if step % 100 == 0 or step == steps - 1:
                avg = sum(losses[-100:]) / len(losses[-100:])
                extra = ""
                if 'router' in info:
                    ri = info['router']
                    ws = " ".join(f"{k}={v:.2f}" for k, v in ri.items() if k.startswith('w_'))
                    extra = f" [{ws}]"
                print(f"    шаг {step:4d}/{steps} | loss={avg:.4f}{extra} | {time.time()-t0:.1f}с")

        final = sum(losses[-50:]) / len(losses[-50:])
        print(f"    Итого: loss {losses[0]:.3f} → {final:.4f}")

        if final < best_loss:
            best_loss = final
            best_mode = mode
            best_orch = orch

    return best_orch, best_mode


# ═══════════════════════════════════════════════════════════════
# Главный скрипт
# ═══════════════════════════════════════════════════════════════

def main():
    random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("  ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ НА INFO_CORPUS")
    print("=" * 70)

    # 1. Токенизатор
    tokenizer = Tokenizer(load_tokenizer())
    print(f"\n  Токенизатор: {'BPE (4096)' if tokenizer.sp else 'char-level'}")

    # 2. Корпус
    print(f"\n  Загрузка корпуса...")
    sequences = load_corpus(tokenizer, block_size=64)
    if not sequences:
        print("  ОШИБКА: нет данных для обучения!")
        return

    # 3. Создание моделей
    print(f"\n{'=' * 70}")
    print(f"  СОЗДАНИЕ МОДЕЛЕЙ")
    print(f"{'=' * 70}")
    models = create_all_models()
    print(f"\n  Всего моделей: {len(models)}")

    # 4. Обучение каждой модели индивидуально
    print(f"\n{'=' * 70}")
    print(f"  ИНДИВИДУАЛЬНОЕ ОБУЧЕНИЕ")
    print(f"{'=' * 70}")

    QUESTIONS = [
        "Что такое энергия",
        "Музыка это",
        "Свет и тьма",
        "Мудрость начинается",
        "Феникс возрождается",
        "Всё течёт",
        "Число пи",
        "Свобода это",
    ]

    trained_losses = {}
    for name, model in models.items():
        # NautilusMoME уже обучена — дообучаем меньше
        # HMoE тоже из чекпоинта — дообучаем
        steps = 300 if name in ('nautilus_mome', 'hmoe') else 500
        lr = 1e-4 if name in ('nautilus_mome', 'hmoe') else 3e-4

        losses = train_model(name, model, sequences, steps=steps, lr=lr)
        trained_losses[name] = losses

        # Генерация после обучения
        print(f"\n  Ответы {name}:")
        for q in QUESTIONS[:4]:
            text = generate(model, tokenizer, q, max_len=60)
            short = text[:90] + "..." if len(text) > 90 else text
            print(f"    «{q}» → {short}")

    # 5. Сравнительная таблица
    print(f"\n{'=' * 70}")
    print(f"  РЕЗУЛЬТАТЫ ИНДИВИДУАЛЬНОГО ОБУЧЕНИЯ")
    print(f"{'=' * 70}")
    print(f"\n  {'Модель':<20s} {'Start':>8s} {'Final':>8s} {'Снижение':>10s}")
    print(f"  {'─' * 50}")
    for name, losses in sorted(trained_losses.items(), key=lambda x: sum(x[1][-50:]) / max(len(x[1][-50:]), 1)):
        if losses:
            start = losses[0]
            final = sum(losses[-50:]) / len(losses[-50:])
            pct = (1 - final / start) * 100
            print(f"  {name:<20s} {start:>8.3f} {final:>8.4f} {pct:>9.1f}%")

    # 6. Grand Orchestrator
    best_orch, best_mode = train_orchestrator_ensemble(
        models, sequences, tokenizer, steps=300, lr=2e-3)

    # 7. Финальный диалог
    print(f"\n{'=' * 70}")
    print(f"  ФИНАЛЬНЫЙ ДИАЛОГ (Grand Orchestrator, режим {best_mode.upper()})")
    print(f"{'=' * 70}")

    for q in QUESTIONS:
        from yijing_transformer.models.grand_orchestrator import GrandOrchestrator
        best_orch.eval()
        ids = tokenizer.encode(q)
        idx = torch.tensor([ids], dtype=torch.long)
        for _ in range(80):
            idx_cond = idx[:, -64:]
            logits, _, _ = best_orch(idx_cond, mode=best_mode)
            logits = logits[:, -1, :] / 0.7
            top_k = 40
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            ch = tokenizer.decode([next_id.item()])
            if ch and ch[-1] in '.!?\n':
                break
        text = tokenizer.decode(idx[0].tolist())
        print(f"\n  Q: «{q}»")
        print(f"  A: {text}")

    # 8. Сохранение чекпоинтов
    ckpt_dir = os.path.join(ROOT, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for name, model in models.items():
        path = os.path.join(ckpt_dir, f'{name}_trained.pt')
        torch.save({
            'model_state': model.state_dict(),
            'name': name,
            'losses': trained_losses.get(name, []),
        }, path)
        print(f"\n  Сохранён: {path}")

    # Сохраняем оркестратор
    orch_path = os.path.join(ckpt_dir, 'grand_orchestrator.pt')
    torch.save({
        'model_state': best_orch.state_dict(),
        'mode': best_mode,
        'model_names': list(models.keys()),
    }, orch_path)
    print(f"  Сохранён: {orch_path}")

    print(f"\n{'=' * 70}")
    print(f"  ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"  Обучено {len(models)} моделей + Grand Orchestrator ({best_mode})")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

import sys, os, random, time, glob
sys.path.insert(0, '/home/user/pro2')
sys.path.insert(0, '/home/user/pro2/yijing_transformer/scripts')
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

random.seed(42); torch.manual_seed(42)
ROOT = '/home/user/pro2'

sp = spm.SentencePieceProcessor()
sp.load(f'{ROOT}/yijing_transformer/bpe_tokenizer.model')

# ═══════ LOAD FULL CORPUS ═══════
print('=' * 65)
print('  ЗАГРУЗКА РАСШИРЕННОГО КОРПУСА')
print('=' * 65)

all_texts = []

# 1. info_corpus (653 KB)
with open(f'{ROOT}/data/info_corpus/combined_corpus.txt', 'r') as f:
    all_texts.append(('info_corpus', f.read()))

# 2. svend4 repos (20 MB)
corpus_dir = f'{ROOT}/data/svend4_corpus'
for cat in ['ai_agents', 'infosystems', 'knowledge', 'algorithms']:
    cat_dir = os.path.join(corpus_dir, cat)
    if not os.path.isdir(cat_dir):
        continue
    for root, dirs, files in os.walk(cat_dir):
        for fn in files:
            if fn.endswith(('.md', '.txt', '.skill', '.rst')):
                fp = os.path.join(root, fn)
                try:
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                        txt = f.read()
                    if len(txt) > 50:
                        all_texts.append((cat, txt))
                except:
                    pass

total_chars = sum(len(t) for _, t in all_texts)
print(f'  Источников: {len(all_texts)}')
print(f'  Символов: {total_chars:,} ({total_chars/1024/1024:.1f} МБ)')

# Tokenize into sequences
block_size = 64
seqs = []
for source, txt in all_texts:
    paragraphs = [p.strip() for p in txt.split('\n\n') if len(p.strip()) > 20]
    for p in paragraphs:
        ids = sp.encode(p)
        if len(ids) >= 10:
            for i in range(0, len(ids) - block_size, block_size // 2):
                s = ids[i:i+block_size+1]
                if len(s) == block_size+1:
                    seqs.append(s)
            if len(ids) <= block_size+1:
                seqs.append((ids + [0]*(block_size+1-len(ids)))[:block_size+1])

random.shuffle(seqs)
print(f'  Обучающих последовательностей: {len(seqs)} (было 4007 на info_corpus)')
print(f'  Увеличение: {len(seqs)/4007:.1f}x')

# ═══════ BUILD MODELS ═══════
print(f'\n{"="*65}')
print(f'  ЗАГРУЗКА МОДЕЛЕЙ')
print(f'{"="*65}')
models = {}

# 1. NautilusMoME (pretrained)
from train_nautilus_mome import NautilusMoME
ckpt = torch.load(f'{ROOT}/yijing_transformer/train_mome_checkpoint.pt', map_location='cpu', weights_only=False)
args = ckpt.get('args', {})
m = NautilusMoME(vocab_size=args.get('vocab_size',4096), d_model=args.get('d_model',128),
                 n_layers=args.get('n_layers',4), n_heads=args.get('n_heads',4),
                 block_size=args.get('block_size',256), d_expert=args.get('d_expert',128),
                 n_experts=args.get('n_experts',6), top_k=args.get('top_k',2))
m.load_state_dict(ckpt['model'], strict=False)
models['nautilus_mome'] = m
print(f'  ✓ nautilus_mome (pretrained)')

# 2-5. New models
from yijing_transformer.models.polyglot import build_polyglot
models['quartet'] = build_polyglot(4096, 128, 2, block_size=256)
print(f'  ✓ quartet')

from yijing_transformer.models.variant3 import Variant3GPT, Variant3Config
models['variant3'] = Variant3GPT(Variant3Config(vocab_size=4096, d_model=128, n_layers=2, n_heads=4, block_size=256))
print(f'  ✓ variant3')

from yijing_transformer.models.nautilus_yijing import NautilusYiJing, NautilusYiJingConfig
models['nautilus_yijing'] = NautilusYiJing(NautilusYiJingConfig(
    vocab_size=4096, d_model=128, block_size=256, n_layers=4, n_heads=4,
    d_expert=64, n_experts=6, top_k=2, dropout=0.05, enable_synth=False))
print(f'  ✓ nautilus_yijing')

from yijing_transformer.models.hierarchical_e2 import HierarchicalE2, E2Config
models['hierarchical_e2'] = HierarchicalE2(E2Config(vocab_size=4096, d_model=128, block_size=256, n_core=2, n_heads=4))
print(f'  ✓ hierarchical_e2')

# 6. YiJingGPT (checkpoint + vocab expand)
from yijing_transformer.models.model import YiJingGPT
ckpt = torch.load(f'{ROOT}/yijing_transformer/train_real_data_checkpoint.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('model', ckpt)
v, d = sd['tok_emb.weight'].shape
class Cfg: pass
cfg = Cfg()
cfg.vocab_size=v; cfg.d_model=d; cfg.block_size=256; cfg.n_layers=4
cfg.n_heads=max(d//64,2); cfg.ffn_mult=4; cfg.dropout=0.05
cfg.use_rope=True; cfg.weight_tying=True; cfg.label_smoothing=0.0
for a in ['use_four_level_pe','use_cubic_pe','use_bidirectional_tri','use_convergence_bridge',
          'use_matrix_grammar','use_abriale','use_nautilus','use_pseudo_rag','use_diff_attn',
          'use_expert_choice','use_six_sources','use_alibi','use_glyph_tokenizer','use_glyph_prior',
          'use_gradient_ckpt','use_gumbel','use_hex_moe','use_swiglu','use_flash_attn',
          'use_bian_gua','use_quadrant_attention']:
    setattr(cfg, a, False)
cfg.bias=False; cfg.n_kv_heads=None; cfg.sliding_window=None; cfg.attention_sinks=0
cfg.rope_base=10000; cfg.rope_scaling=None; cfg.rope_scaling_factor=1.0
cfg.quantizer_type='factored6'; cfg.quant_total_dim=6; cfg.quant_dim_schedule=None
cfg.quant_group_dim=6; cfg.multi_scale_quant=False; cfg.temp=1.0; cfg.adaptive_temp=False
cfg.commitment_weight=0.25; cfg.head_dim=d//cfg.n_heads; cfg.n_experts=0; cfg.moe_top_k=2
cfg.hex_strength=0.1; cfg.gate_init_bias=0.0; cfg.total_steps=10000; cfg.token_merge_ratio=0.0
cfg.prefix_len=0; cfg.mtp_n_future=0; cfg.ffn_hidden=d*cfg.ffn_mult; cfg.distill_temp=2.0
cfg.pseudo_rag_distill_weight=0.1; cfg.curriculum_strategy_geo='linear'
cfg.curriculum_target_strength=1.0; cfg.curriculum_warmup_fraction=0.1
cfg.convergence_n_clusters=64; cfg.convergence_window_size=4; cfg.convergence_stride=2
cfg.convergence_compose_layers=1; cfg.convergence_n_heads=4
cfg.abriale_d_event=64; cfg.abriale_n_heads=4; cfg.abriale_arity=2; cfg.abriale_n_rules=64
cfg.abriale_n_hits=4; cfg.abriale_n_alternatives=2; cfg.abriale_n_event_types=8
cfg.abriale_balance_weight=0.01; cfg.nautilus_chambers='all'; cfg.nautilus_init_scale=0.01
cfg.nautilus_warmup_steps=2000; cfg.nautilus_mode='sequential'
cfg.matrix_grammar_rows=8; cfg.matrix_grammar_cols=8; cfg.matrix_grammar_heads=4
m = YiJingGPT(cfg); m.load_state_dict(sd, strict=False)
if v < 4096:
    oe=m.tok_emb.weight.data; ne=nn.Embedding(4096,d); nn.init.normal_(ne.weight,std=0.02)
    ne.weight.data[:v]=oe; m.tok_emb=ne
    oh=m.head; nh=nn.Linear(d,4096,bias=oh.bias is not None); nn.init.normal_(nh.weight,std=0.02)
    if nh.bias is not None: nn.init.zeros_(nh.bias)
    nh.weight.data[:v]=oh.weight.data
    if oh.bias is not None and nh.bias is not None: nh.bias.data[:v]=oh.bias.data
    m.head=nh
models['yijing'] = m
print(f'  ✓ yijing (checkpoint, vocab {v}→4096)')

# 7. HMoE (checkpoint + vocab expand)
from yijing_transformer.models.hierarchical_moe import HMoEConfig, HierarchicalMoEFFN
from yijing_transformer.models.geometry.routing import ArchetypalInterlingua
ckpt = torch.load(f'{ROOT}/hmoe_fixed_joint.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('model_state', ckpt)
mc=dict(vocab_size=256,block_size=64,d_model=128,n_heads=4,n_layers=4,ffn_mult=4,
        hamming_lambda=0.15,uncertainty_budget=0.25,dropout=0.1,
        use_domain_routing=False,use_hierarchical_moe=True)
m = Variant3GPT(Variant3Config(**mc))
hc = HMoEConfig(d_model=128, use_multiscale=True, use_hex_tier=False)
for b in m.blocks:
    if hasattr(b,'hmoe'): b.hmoe = HierarchicalMoEFFN(hc)
def _mk(il):
    _o=il.forward
    def _f(so,ch): return _o(ch,so),(il.get_interlingua_loss() if hasattr(il,'get_interlingua_loss') else 0.0)
    return _f
for b in m.blocks:
    if hasattr(b,'interlingua'):
        il=ArchetypalInterlingua(d_model=128,n_sources=2,n_archetypes=64,uncertainty_budget=0.25)
        il.forward=_mk(il); b.interlingua=il
m.load_state_dict(sd, strict=False)
oe=m.tok_emb.weight.data; ne=nn.Embedding(4096,128); nn.init.normal_(ne.weight,std=0.02)
ne.weight.data[:256]=oe; m.tok_emb=ne
oh=m.head; nh=nn.Linear(128,4096,bias=oh.bias is not None); nn.init.normal_(nh.weight,std=0.02)
if nh.bias is not None: nn.init.zeros_(nh.bias)
nh.weight.data[:256]=oh.weight.data
if oh.bias is not None and nh.bias is not None: nh.bias.data[:256]=oh.bias.data
m.head=nh
models['hmoe'] = m
print(f'  ✓ hmoe (checkpoint, vocab 256→4096)')

print(f'\n  Итого: {len(models)} моделей, {sum(sum(p.numel() for p in m.parameters()) for m in models.values()):,} параметров')

# ═══════ TRAINING ═══════
QUESTIONS = [
    'Что такое энергия',
    'Музыка это',
    'Свет и тьма',
    'Мудрость начинается',
    'Феникс возрождается',
    'Свобода это',
    'Искусственный интеллект',
    'Алгоритм это',
    'Информационная система',
    'Архетип',
]

def generate(model, prompt, max_len=100, temp=0.75):
    model.eval()
    ids = sp.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_len):
            ic = idx[:, -block_size:]
            try: result = model(ic)
            except: break
            logits = (result[0] if isinstance(result, tuple) else result)[:, -1, :] / temp
            tk = 50; v, _ = torch.topk(logits, min(tk, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            nid = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nid], dim=1)
            ch = sp.decode([nid.item()])
            if ch and ch[-1] in '.!?\n': break
    return sp.decode(idx[0].tolist())

def train_one(name, model, steps, lr):
    model.train()
    for p in model.parameters(): p.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_p = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr*0.05)
    t0 = time.time()
    losses = []
    for step in range(steps):
        idxs = [random.randint(0, len(seqs)-1) for _ in range(16)]
        x = torch.tensor([seqs[i][:block_size] for i in idxs], dtype=torch.long)
        y = torch.tensor([seqs[i][1:block_size+1] for i in idxs], dtype=torch.long)
        try: result = model(x, targets=y)
        except TypeError:
            try: result = model(x, y)
            except: result = model(x)
        if isinstance(result, tuple) and len(result) >= 2:
            logits, loss = result[0], result[1]
        else:
            logits = result[0] if isinstance(result, tuple) else result
            loss = None
        if loss is None or (hasattr(loss,'item') and loss.item() == 0):
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sch.step()
        losses.append(loss.item())
        if step % 1000 == 0 or step == steps-1:
            avg = sum(losses[-1000:])/len(losses[-1000:])
            print(f'    шаг {step:5d}/{steps} | loss={avg:.4f} | best={min(losses):.4f} | {(step+1)/(time.time()-t0):.1f} ш/с')
    final = sum(losses[-200:])/len(losses[-200:])
    print(f'    Итого: {time.time()-t0:.0f}с, {losses[0]:.2f} → {final:.4f} ({(1-final/losses[0])*100:.0f}%)')
    return losses

# Train config - more steps with bigger corpus
config = {
    'nautilus_mome':   (3000, 2e-4),
    'variant3':        (8000, 5e-4),
    'nautilus_yijing': (8000, 5e-4),
    'yijing':          (5000, 3e-4),
    'hmoe':            (5000, 3e-4),
    'quartet':         (8000, 5e-4),
    'hierarchical_e2': (8000, 5e-4),
}

results = {}
for name, model in models.items():
    steps, lr = config[name]
    n_p = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*65}')
    print(f'  {name} ({n_p:,} п.) — {steps} шагов, lr={lr}')
    print(f'{"="*65}')
    losses = train_one(name, model, steps, lr)
    results[name] = losses

# ═══════ RESULTS ═══════
print(f'\n{"="*65}')
print(f'  СВОДНАЯ ТАБЛИЦА (полный корпус 20 МБ)')
print(f'{"="*65}')
print(f'\n  {"Модель":<20s} {"Шагов":>6s} {"Start":>7s} {"Final":>7s} {"Снижение":>9s}')
print(f'  {"─"*52}')
ranking = sorted(results.items(), key=lambda x: sum(x[1][-200:])/len(x[1][-200:]))
for name, losses in ranking:
    s = losses[0]; f = sum(losses[-200:])/len(losses[-200:])
    print(f'  {name:<20s} {len(losses):>6d} {s:>7.2f} {f:>7.3f} {(1-f/s)*100:>8.1f}%')

# ═══════ DIALOGUE — TOP 3 ═══════
for rank, (name, losses) in enumerate(ranking[:3], 1):
    model = models[name]
    f = sum(losses[-200:])/len(losses[-200:])
    print(f'\n{"="*65}')
    print(f'  #{rank} {name} (loss={f:.3f})')
    print(f'{"="*65}')
    for q in QUESTIONS:
        t = generate(model, q, max_len=120)
        short = t[:150] + '...' if len(t) > 150 else t
        print(f'\n  Q: «{q}»')
        print(f'  A: {short}')

# Save checkpoints
ckpt_dir = f'{ROOT}/checkpoints'
os.makedirs(ckpt_dir, exist_ok=True)
for name, model in models.items():
    path = os.path.join(ckpt_dir, f'{name}_full.pt')
    torch.save({'model_state': model.state_dict(), 'name': name,
                'corpus': 'info_corpus + svend4 (20MB)',
                'steps': len(results.get(name, [])),
                'final_loss': sum(results[name][-200:])/len(results[name][-200:]) if name in results else 0,
                'losses': results.get(name, [])[-100:]}, path)
print(f'\nЧекпоинты сохранены в {ckpt_dir}/')

print(f'\n{"="*65}')
print(f'  ОБУЧЕНИЕ НА ПОЛНОМ КОРПУСЕ ЗАВЕРШЕНО')
print(f'{"="*65}')

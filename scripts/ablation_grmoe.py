"""Controlled ablation: does GrMoE / LatFormer / SIPS actually help?

Same backbone, same data, same seed, same steps. Held-out validation set.
Variants:
  A. dense       — standard causal MHA + dense FFN (no geometry, no MoE)
  B. linear_moe  — standard MHA + linear top-2 router MoE (Switch-style)
  C. grmoe       — standard MHA + Q6 Grassmannian router MoE
  D. grmoe_lat   — EnhancedBianGuaAttention (LatFormer+SIPS) + Q6 GrMoE
"""
import sys, os, random, time, json, math
sys.path.insert(0, '/home/user/pro2')
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

from yijing_transformer.models.geometry.advanced_routing import (
    Q6GrassmannianRouter, EnhancedBianGuaAttention,
)

ROOT = os.environ.get('PRO2_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('ABL_DIR') or os.path.join(ROOT, 'experiments')
os.makedirs(OUT, exist_ok=True)
STEPS = int(os.environ.get('ABL_STEPS', 3000))
BATCH = 16
BLOCK = 64
LR = 5e-4
D, H, L = 128, 4, 3
N_EXP, D_EXP, TOPK = 6, 128, 2
VAL_EVERY = 500
SEED = 1234

sp = spm.SentencePieceProcessor()
sp.load(f'{ROOT}/yijing_transformer/bpe_tokenizer.model')

# ─── data ───
def load_corpus():
    texts = []
    info_path = f'{ROOT}/data/info_corpus/combined_corpus.txt'
    if os.path.exists(info_path):
        with open(info_path) as f:
            texts.append(f.read())
    for cat in ['ai_agents', 'infosystems', 'knowledge', 'algorithms']:
        d = f'{ROOT}/data/svend4_corpus/{cat}'
        if not os.path.isdir(d):
            continue
        for r, _, files in os.walk(d):
            for fn in files:
                if fn.endswith(('.md', '.txt', '.skill', '.rst')):
                    try:
                        with open(os.path.join(r, fn), encoding='utf-8', errors='ignore') as f:
                            t = f.read()
                        if len(t) > 50:
                            texts.append(t)
                    except OSError:
                        pass
    seqs = []
    for t in texts:
        for p in (pp.strip() for pp in t.split('\n\n') if len(pp.strip()) > 20):
            ids = sp.encode(p)
            if len(ids) < 10:
                continue
            for i in range(0, len(ids) - BLOCK, BLOCK // 2):
                s = ids[i:i + BLOCK + 1]
                if len(s) == BLOCK + 1:
                    seqs.append(s)
            if len(ids) <= BLOCK + 1:
                seqs.append((ids + [0] * (BLOCK + 1 - len(ids)))[:BLOCK + 1])
    return seqs

random.seed(SEED)
seqs = load_corpus()
random.shuffle(seqs)
n_val = 2048
val_seqs = seqs[:n_val]
train_seqs = seqs[n_val:]
print(f'train={len(train_seqs)}  val={len(val_seqs)}')

def batch_from(pool, idxs):
    x = torch.tensor([pool[i][:BLOCK] for i in idxs], dtype=torch.long)
    y = torch.tensor([pool[i][1:BLOCK + 1] for i in idxs], dtype=torch.long)
    return x, y

# fixed train batch order shared by all variants
rng = random.Random(SEED)
train_order = [[rng.randint(0, len(train_seqs) - 1) for _ in range(BATCH)] for _ in range(STEPS)]
val_batches = [batch_from(val_seqs, list(range(i, i + BATCH))) for i in range(0, n_val, BATCH)]

# ─── components ───
class StdAttention(nn.Module):
    def __init__(self, d, h, block):
        super().__init__()
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(block, block)).view(1, 1, block, block))
    def forward(self, x):
        B, T, Dm = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.h, self.hd).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        s = s.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        o = F.softmax(s, -1) @ v
        return self.out(o.transpose(1, 2).reshape(B, T, Dm))

def make_expert(d, de):
    return nn.Sequential(nn.Linear(d, de), nn.GELU(), nn.Linear(de, d))

class LinearRouter(nn.Module):
    """Switch-style top-k router with load-balancing aux loss."""
    def __init__(self, d, n, k):
        super().__init__()
        self.n, self.k = n, k
        self.w = nn.Linear(d, n, bias=False)
        self.register_buffer('expert_load_ema', torch.zeros(n))
    def forward(self, x):
        logits = self.w(x)
        probs = F.softmax(logits, -1)
        tv, ti = torch.topk(probs, self.k, -1)
        w = torch.zeros_like(probs).scatter(2, ti, tv / tv.sum(-1, keepdim=True))
        frac = w.sum((0, 1)) / (x.shape[0] * x.shape[1])
        pmean = probs.mean((0, 1))
        aux = (frac * pmean).sum() * self.n
        with torch.no_grad():
            self.expert_load_ema.mul_(0.99).add_(0.01 * frac)
        return w, ti, aux

class MoEFFN(nn.Module):
    def __init__(self, d, router):
        super().__init__()
        self.router = router
        self.experts = nn.ModuleList([make_expert(d, D_EXP) for _ in range(N_EXP)])
    def forward(self, x):
        w, _, aux = self.router(x)
        outs = torch.stack([e(x) for e in self.experts], 2)
        return (outs * w.unsqueeze(-1)).sum(2), aux

class DenseFFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        # param-matched-ish: 2 experts' worth of compute active in MoE (top-2)
        self.ff = make_expert(d, D_EXP * 2)
    def forward(self, x):
        return self.ff(x), x.new_zeros(())

class Block(nn.Module):
    def __init__(self, attn, ffn):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn, self.ffn = attn, ffn
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        f, aux = self.ffn(self.n2(x))
        return x + f, aux

class LM(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.tok = nn.Embedding(4096, D)
        self.pos = nn.Embedding(256, D)
        blocks = []
        for i in range(L):
            if variant == 'dense':
                attn, ffn = StdAttention(D, H, 256), DenseFFN(D)
            elif variant == 'linear_moe':
                attn, ffn = StdAttention(D, H, 256), MoEFFN(D, LinearRouter(D, N_EXP, TOPK))
            elif variant == 'grmoe':
                attn = StdAttention(D, H, 256)
                ffn = MoEFFN(D, Q6GrassmannianRouter(D, N_EXP, subspace_dim=4, top_k=TOPK, latent_dim=32))
            elif variant == 'grmoe_lat':
                attn = EnhancedBianGuaAttention(D, H, 256, dropout=0.0, use_latformer=True)
                ffn = MoEFFN(D, Q6GrassmannianRouter(D, N_EXP, subspace_dim=4, top_k=TOPK, latent_dim=32))
            else:
                raise ValueError(variant)
            blocks.append(Block(attn, ffn))
        self.blocks = nn.ModuleList(blocks)
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, 4096, bias=False)
    def forward(self, x, y=None):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        aux = h.new_zeros(())
        for b in self.blocks:
            h, a = b(h)
            aux = aux + a
        logits = self.head(self.lnf(h))
        if y is None:
            return logits
        ce = F.cross_entropy(logits.reshape(-1, 4096), y.reshape(-1), ignore_index=0)
        return logits, ce, aux

@torch.no_grad()
def evaluate(m):
    m.eval()
    tot, n = 0.0, 0
    for x, y in val_batches:
        _, ce, _ = m(x, y)
        tot += ce.item(); n += 1
    m.train()
    return tot / n

def routers_of(m):
    return [b.ffn.router for b in m.blocks if hasattr(b.ffn, 'router')]

def run(variant):
    torch.manual_seed(SEED)
    m = LM(variant)
    n_p = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR * 0.05)
    print(f'\n=== {variant}  ({n_p:,} params) ===')
    hist = {'train': [], 'val': [], 'params': n_p}
    t0 = time.time()
    window = []
    for step in range(STEPS):
        x, y = batch_from(train_seqs, train_order[step])
        _, ce, aux = m(x, y)
        loss = ce + 0.01 * aux
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        window.append(ce.item())
        if (step + 1) % VAL_EVERY == 0 or step == STEPS - 1:
            v = evaluate(m)
            tr = sum(window[-VAL_EVERY:]) / len(window[-VAL_EVERY:])
            hist['train'].append((step + 1, tr)); hist['val'].append((step + 1, v))
            print(f'  step {step+1:5d}  train={tr:.4f}  val={v:.4f}  ppl={math.exp(v):.1f}  {(step+1)/(time.time()-t0):.1f} st/s')
    hist['time_s'] = time.time() - t0
    loads = []
    for r in routers_of(m):
        loads.append([round(v, 3) for v in r.expert_load_ema.tolist()])
    hist['expert_load'] = loads
    if variant.startswith('grmoe'):
        hist['geo_gate'] = [round(torch.sigmoid(r.geo_gate).item(), 3) for r in routers_of(m)]
        hist['lambda'] = [round(F.softplus(r.grass_router.log_lambda).mean().item(), 3) for r in routers_of(m)]
    # sample generations
    m.eval()
    gens = {}
    for prompt in ['Что такое мудрость', 'Информационная система', 'Алгоритм поиска']:
        ids = torch.tensor([sp.encode(prompt)])
        with torch.no_grad():
            for _ in range(40):
                lg = m(ids[:, -BLOCK:])[:, -1, :] / 0.7
                for t in set(ids[0].tolist()[-30:]):
                    lg[0, t] /= 1.3
                v, _ = torch.topk(lg, 40)
                lg[lg < v[:, [-1]]] = float('-inf')
                nxt = torch.multinomial(F.softmax(lg, -1), 1)
                ids = torch.cat([ids, nxt], 1)
                if sp.decode([nxt.item()]).endswith(('.', '!', '?', '\n')):
                    break
        gens[prompt] = sp.decode(ids[0].tolist())
    hist['gens'] = gens
    return hist

results = {}
VARIANTS = os.environ.get('ABL_VARIANTS', 'dense,linear_moe,grmoe,grmoe_lat').split(',')
OUT_JSON = os.environ.get('ABL_OUT', 'ablation_results.json')
for v in VARIANTS:
    results[v] = run(v)
    with open(os.path.join(OUT, OUT_JSON), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

print('\n' + '=' * 72)
print(f'{"variant":<12} {"params":>10} {"train":>8} {"val":>8} {"val ppl":>8} {"min load":>9} {"max load":>9} {"time":>7}')
for v, h in results.items():
    tr = h['train'][-1][1]; va = h['val'][-1][1]
    loads = [x for layer in h['expert_load'] for x in layer] or [1 / N_EXP]
    print(f'{v:<12} {h["params"]:>10,} {tr:>8.3f} {va:>8.3f} {math.exp(va):>8.1f} {min(loads):>9.3f} {max(loads):>9.3f} {h["time_s"]:>6.0f}s')
for v, h in results.items():
    print(f'\n--- {v} ---')
    if 'geo_gate' in h:
        print(f'  geo_gate={h["geo_gate"]}  lambda={h["lambda"]}')
    for p, g in h['gens'].items():
        print(f'  {g[:160]}')

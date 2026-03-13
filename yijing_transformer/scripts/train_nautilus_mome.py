#!/usr/bin/env python3
"""
NautilusMoME — Mixture of Micro-Experts with BPE tokenizer.

Architecture:
  ┌─────────────┐
  │   ROUTER     │  ← decides which experts to activate (top-2)
  └──────┬──────┘
  ┌──────▼──────┐
  │  CORE (4L)   │  ← always active, learns general principles
  └──┬──┬──┬──┬─┘
     │  │  │  │
  ┌──▼──▼──▼──▼──┐
  │ MICRO-EXPERTS  │  ← 6 domain specialists (~50K each)
  │ MATH CODE HUM  │     can be trained/frozen independently
  │ SYS  RECON INFO│
  └───────┬───────┘
  ┌───────▼───────┐
  │ NAUTILUS BRIDGE │  ← hierarchical aggregation (from NautilusHierarchy)
  └───────────────┘

Key innovations:
  - BPE tokenizer (sentencepiece, vocab=4096) instead of byte-level
  - Semantic experts: each domain = separate trainable module
  - Sparse activation: only 2/6 experts active per token
  - Modular: experts can be added/removed/retrained independently
  - NautilusHierarchy as bridge between core and expert outputs

Training phases:
  Phase 0: Train BPE tokenizer on all data
  Phase 1: Train Core + all Experts end-to-end
  Phase 2: Fine-tune individual experts on domain-specific data
  Phase 3: Fine-tune Router on mixed data

Usage:
  python train_nautilus_mome.py                    # Full pipeline
  python train_nautilus_mome.py --phase 1          # Skip tokenizer training
  python train_nautilus_mome.py --phase 2          # Expert fine-tuning only
  python train_nautilus_mome.py --resume ckpt.pt   # Resume training
"""

import sys
import os
import math
import time
import json
import random
import argparse
import glob as glob_module
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# ==================== Configuration ====================

EXPERT_DOMAINS = {
    'MATH': {
        'name': 'Mathematical Structures',
        'repos': ['meta', 'data2', 'pro2'],
        'description': 'Math, formulas, algorithms, hexagrams, transformers',
    },
    'CODE': {
        'name': 'Software Engineering',
        'repos': ['daten3', 'daten2', 'data20'],
        'description': 'TypeScript, Flask, React, full-stack, KMS',
    },
    'HUMAN': {
        'name': 'Humanitarian Knowledge',
        'repos': ['info3', 'daten22', 'info'],
        'description': 'Ethics, archetypes, MBTI, behavioral formulas',
    },
    'SYSTEM': {
        'name': 'System Architecture',
        'repos': ['info7', 'daten', 'universal-file-storage-mcp'],
        'description': 'AI orchestration, DevOps, MCP, containers',
    },
    'RECON': {
        'name': 'Pattern Recognition',
        'repos': ['meta2'],
        'description': 'Document reconstruction, OCR, puzzle algorithms',
    },
    'INFO': {
        'name': 'Information Management',
        'repos': ['info1', 'info4', 'info5', 'daten11', 'data30', 'in4'],
        'description': 'Knowledge bases, catalogs, search, automation',
    },
}

ALL_REPO_DIRS = [
    # Previously used (Tier 0)
    '/tmp/meta2', '/tmp/meta', '/tmp/data2', '/tmp/info1',
    '/tmp/info3', '/tmp/info7', '/tmp/info4', '/tmp/info',
    '/tmp/info2', '/tmp/info5',
    # New repos (Tier 1 + 2)
    '/tmp/daten3', '/tmp/data20', '/tmp/data30', '/tmp/daten',
    '/tmp/pro2', '/tmp/daten22', '/tmp/daten2', '/tmp/daten11',
    '/tmp/universal-file-storage-mcp', '/tmp/in4',
]

TEXT_EXTENSIONS = {
    '.txt', '.md', '.py', '.json', '.yaml', '.yml', '.js', '.jsx',
    '.ts', '.tsx', '.html', '.htm', '.css', '.csv', '.xml', '.sql',
    '.sh', '.bat', '.cfg', '.ini', '.toml', '.skill', '.gitignore',
    '.kt', '.java', '.rs', '.go', '.rb', '.php', '.vue', '.svelte',
    '.dockerfile', '.env.example', '.conf',
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(BASE_DIR, 'train_mome_results.json')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'train_mome_checkpoint.pt')
TOKENIZER_MODEL = os.path.join(BASE_DIR, 'bpe_tokenizer.model')
TOKENIZER_VOCAB = os.path.join(BASE_DIR, 'bpe_tokenizer.vocab')
RAW_TEXT_PATH = '/tmp/mome_training_text.txt'
DOMAIN_DIR = '/tmp/mome_domains/'


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


# ==================== Data Collection ====================

def collect_text_from_repos(repo_dirs=None):
    """Collect all text files from cloned repos."""
    if repo_dirs is None:
        repo_dirs = ALL_REPO_DIRS

    all_text = []
    total_bytes = 0
    total_files = 0

    for repo_dir in repo_dirs:
        if not os.path.isdir(repo_dir):
            print(f"  SKIP (not found): {repo_dir}")
            continue

        repo_name = os.path.basename(repo_dir)
        repo_bytes = 0
        repo_files = 0

        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', '__pycache__', '.venv', 'venv')]

            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext not in TEXT_EXTENSIONS and ext != '':
                    continue
                # Skip binary-looking files
                if f.endswith(('.min.js', '.min.css', '.map', '.lock')):
                    continue

                path = os.path.join(root, f)
                try:
                    # Skip large files (likely generated/data)
                    if os.path.getsize(path) > 500_000:
                        continue
                    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                        content = fh.read()
                    if len(content.strip()) < 10:
                        continue
                    # Skip binary content (null bytes, high non-printable ratio)
                    content = content.replace('\x00', '')
                    if sum(1 for c in content[:1000] if ord(c) < 32 and c not in '\n\r\t') > 50:
                        continue
                    # File header for context
                    header = f"### {repo_name}/{os.path.relpath(path, repo_dir)}\n"
                    all_text.append(header + content + "\n")
                    repo_bytes += len(content)
                    repo_files += 1
                except Exception:
                    continue

        total_bytes += repo_bytes
        total_files += repo_files
        print(f"  {repo_name}: {repo_files} files, {repo_bytes:,} bytes")

    print(f"  TOTAL: {total_files} files, {total_bytes:,} bytes")
    return all_text


def prepare_domain_data():
    """Split collected text into domain-specific files for expert training."""
    os.makedirs(DOMAIN_DIR, exist_ok=True)

    # Map repo → domain
    repo_to_domain = {}
    for domain_key, domain_info in EXPERT_DOMAINS.items():
        for repo in domain_info['repos']:
            repo_to_domain[repo] = domain_key

    domain_texts = defaultdict(list)

    for repo_dir in ALL_REPO_DIRS:
        if not os.path.isdir(repo_dir):
            continue
        repo_name = os.path.basename(repo_dir)
        domain = repo_to_domain.get(repo_name, 'INFO')  # default to INFO

        texts = collect_text_from_repos([repo_dir])
        domain_texts[domain].extend(texts)

    print("\n  Domain data distribution:")
    for domain, texts in domain_texts.items():
        text_blob = '\n'.join(texts)
        domain_path = os.path.join(DOMAIN_DIR, f'{domain}.txt')
        with open(domain_path, 'w', encoding='utf-8') as f:
            f.write(text_blob)
        print(f"    {domain}: {len(texts)} files, {len(text_blob):,} chars")

    return domain_texts


# ==================== BPE Tokenizer ====================

def train_bpe_tokenizer(vocab_size=4096):
    """Train sentencepiece BPE tokenizer on all collected text."""
    if os.path.exists(TOKENIZER_MODEL):
        print(f"  BPE tokenizer already exists: {TOKENIZER_MODEL}")
        return load_tokenizer()

    print("  Collecting text for tokenizer training...")
    texts = collect_text_from_repos()
    full_text = '\n'.join(texts)

    with open(RAW_TEXT_PATH, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"  Training BPE tokenizer (vocab={vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=RAW_TEXT_PATH,
        model_prefix=os.path.join(BASE_DIR, 'bpe_tokenizer'),
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9999,
        num_threads=4,
        max_sentence_length=16384,
        shuffle_input_sentence=True,
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # BPE-specific
        byte_fallback=True,  # handle unknown bytes
        split_digits=True,
        split_by_unicode_script=True,
        treat_whitespace_as_suffix=False,
    )

    print(f"  Tokenizer saved: {TOKENIZER_MODEL}")
    return load_tokenizer()


def load_tokenizer():
    """Load trained sentencepiece tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    return sp


def encode_data(sp, texts, val_fraction=0.1):
    """Encode text with BPE tokenizer, split train/val."""
    full_text = '\n'.join(texts)

    # Shuffle paragraphs
    paragraphs = full_text.split('\n\n')
    random.seed(42)
    random.shuffle(paragraphs)
    full_text = '\n\n'.join(paragraphs)

    # Encode with BPE
    token_ids = sp.encode(full_text)
    data = torch.tensor(token_ids, dtype=torch.long)

    split_idx = int(len(data) * (1 - val_fraction))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"  BPE encoded: {len(data):,} tokens (vocab={sp.get_piece_size()})")
    print(f"  Train: {len(train_data):,}, Val: {len(val_data):,}")
    print(f"  Compression ratio: {len(full_text.encode('utf-8'))/len(data):.1f} bytes/token")

    return train_data, val_data


def encode_domain_data(sp, domain_dir=DOMAIN_DIR, val_fraction=0.1, max_tokens=2_000_000):
    """Encode domain-specific data for expert fine-tuning.
    Caps each domain at max_tokens to prevent OOM."""
    domain_data = {}
    for domain_key in EXPERT_DOMAINS:
        path = os.path.join(domain_dir, f'{domain_key}.txt')
        if not os.path.exists(path):
            continue
        # Read in chunks to limit memory
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read(max_tokens * 4)  # ~4 chars per BPE token estimate
        tokens = sp.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        data = torch.tensor(tokens, dtype=torch.long)
        split_idx = int(len(data) * (1 - val_fraction))
        domain_data[domain_key] = {
            'train': data[:split_idx],
            'val': data[split_idx:],
        }
        print(f"    {domain_key}: {len(data):,} tokens (capped={len(tokens)==max_tokens})")
        del text, tokens  # free memory
    return domain_data


# ==================== Model Architecture ====================

class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(self, d_model, n_heads, dropout=0.05):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        h = self.ln1(x)
        T = h.size(1)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x


class MicroExpert(nn.Module):
    """A small domain-specific expert module (~50K params).

    Each expert has:
    - A domain-specific FFN (bottleneck adapter)
    - A domain-specific gate that modulates contribution
    - Can be frozen/unfrozen independently
    """

    def __init__(self, d_model, d_expert=128, dropout=0.05):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_expert),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_expert, d_expert),
            nn.GELU(),
            nn.Linear(d_expert, d_model),
            nn.Dropout(dropout),
        )
        # Learnable domain gate (starts small, grows during training)
        self.gate_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """Returns expert contribution (delta, not full output)."""
        return self.adapter(x) * self.gate_scale


class ExpertRouter(nn.Module):
    """Routes tokens to top-k micro-experts.

    Uses learned routing with load balancing loss.
    With BPE tokens, the router can understand semantic context.
    """

    def __init__(self, d_model, n_experts, top_k=2, temperature=1.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.temperature = temperature

        # Two-layer router for better context understanding
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_experts),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            expert_weights: (B, T, n_experts) — sparse, top-k non-zero
            aux_loss: load balancing loss
        """
        logits = self.router(x) / self.temperature  # (B, T, n_experts)

        # Top-k selection
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Scatter to full expert dimension
        expert_weights = torch.zeros_like(logits)
        expert_weights.scatter_(-1, top_k_indices, top_k_weights)

        # Load balancing auxiliary loss
        # Encourage uniform expert utilization
        avg_routing = expert_weights.mean(dim=(0, 1))  # (n_experts,)
        target = torch.ones_like(avg_routing) / self.n_experts
        aux_loss = F.mse_loss(avg_routing, target) * self.n_experts

        return expert_weights, aux_loss


class NautilusBridge(nn.Module):
    """Simplified bridge that merges expert outputs hierarchically.

    Inspired by NautilusHierarchy but adapted for MoME:
    - Takes weighted expert outputs
    - Applies hierarchical merging (pairs → groups → all)
    - Uses residual gating for stability
    """

    def __init__(self, d_model, n_experts):
        super().__init__()
        self.n_experts = n_experts

        # Pairwise merge (experts 0+1, 2+3, 4+5 → 3 groups)
        n_pairs = (n_experts + 1) // 2
        self.pair_merge = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
            ) for _ in range(n_pairs)
        ])

        # Global merge (3 groups → 1)
        self.global_merge = nn.Sequential(
            nn.Linear(d_model * n_pairs, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Residual gate (starts small for training stability)
        self.residual_gate = nn.Parameter(torch.tensor(0.1))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, core_hidden, expert_outputs, expert_weights):
        """
        Args:
            core_hidden: (B, T, d_model) — output from core layers
            expert_outputs: list of (B, T, d_model) — one per expert
            expert_weights: (B, T, n_experts) — routing weights
        Returns:
            merged: (B, T, d_model)
        """
        B, T, D = core_hidden.shape

        # Weight expert outputs
        weighted = []
        for i, exp_out in enumerate(expert_outputs):
            w = expert_weights[:, :, i:i+1]  # (B, T, 1)
            weighted.append(exp_out * w)

        # Pairwise merge
        pair_outputs = []
        for i in range(0, len(weighted), 2):
            if i + 1 < len(weighted):
                pair_in = torch.cat([weighted[i], weighted[i+1]], dim=-1)
            else:
                pair_in = torch.cat([weighted[i], torch.zeros_like(weighted[i])], dim=-1)
            pair_outputs.append(self.pair_merge[i // 2](pair_in))

        # Global merge
        global_in = torch.cat(pair_outputs, dim=-1)
        # Pad if needed
        expected = self.global_merge[0].in_features
        if global_in.size(-1) < expected:
            pad = torch.zeros(B, T, expected - global_in.size(-1), device=global_in.device)
            global_in = torch.cat([global_in, pad], dim=-1)

        merged = self.global_merge(global_in)

        # Residual connection with gate
        output = core_hidden + self.residual_gate * self.ln(merged)
        return output


class NautilusMoME(nn.Module):
    """Nautilus Mixture of Micro-Experts Language Model.

    Architecture:
        Input → Embedding
          → Core Layers (first half)
          → Router (selects top-2 experts)
          → Micro-Experts (6 domain specialists)
          → NautilusBridge (hierarchical merge)
          → Core Layers (second half)
          → LM Head → Output
    """

    EXPERT_NAMES = ['MATH', 'CODE', 'HUMAN', 'SYSTEM', 'RECON', 'INFO']

    def __init__(self, vocab_size=4096, d_model=192, n_layers=4, n_heads=6,
                 block_size=512, d_expert=128, n_experts=6, top_k=2,
                 dropout=0.05):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.n_experts = n_experts
        self.vocab_size = vocab_size

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        # Core transformer layers (split: first half → experts → second half)
        n_first = n_layers // 2
        n_second = n_layers - n_first
        self.core_first = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_first)
        ])
        self.core_second = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_second)
        ])

        # Router
        self.router = ExpertRouter(d_model, n_experts, top_k=top_k)

        # Micro-Experts
        self.experts = nn.ModuleDict({
            name: MicroExpert(d_model, d_expert, dropout)
            for name in self.EXPERT_NAMES[:n_experts]
        })

        # NautilusBridge
        self.bridge = NautilusBridge(d_model, n_experts)

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (embedding ↔ head)
        self.head.weight = self.tok_emb.weight

        # Proper initialization for stable start
        self._init_weights()
        self._step = 0

    def _init_weights(self):
        """Initialize weights for stable training start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def set_step(self, step):
        self._step = step

    def get_expert_params(self, expert_name):
        """Get parameters for a specific expert (for selective training)."""
        if expert_name in self.experts:
            return self.experts[expert_name].parameters()
        return iter([])

    def freeze_core(self):
        """Freeze core layers (for expert fine-tuning)."""
        for p in self.core_first.parameters():
            p.requires_grad = False
        for p in self.core_second.parameters():
            p.requires_grad = False
        for p in self.tok_emb.parameters():
            p.requires_grad = False
        for p in self.pos_emb.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def freeze_all_experts_except(self, expert_name):
        """Freeze all experts except the specified one."""
        for name, expert in self.experts.items():
            for p in expert.parameters():
                p.requires_grad = (name == expert_name)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)

        # Core first half
        for layer in self.core_first:
            x = layer(x)

        # Route to experts
        expert_weights, aux_loss = self.router(x)

        # Run all experts (sparse: only top-k have non-zero weights)
        expert_outputs = []
        expert_names = self.EXPERT_NAMES[:self.n_experts]
        for i, name in enumerate(expert_names):
            # Only compute if any token routes to this expert
            mask = expert_weights[:, :, i].sum() > 0
            if mask:
                exp_out = self.experts[name](x)
            else:
                exp_out = torch.zeros_like(x)
            expert_outputs.append(exp_out)

        # Bridge merges expert outputs with core
        x = self.bridge(x, expert_outputs, expert_weights)

        # Core second half
        for layer in self.core_second:
            x = layer(x)

        # LM head
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Add routing auxiliary loss (small weight)
            loss = loss + 0.01 * aux_loss

        return logits, loss, {'aux_loss': aux_loss.item(), 'routing': expert_weights.detach()}


# ==================== Training Utilities ====================

def get_batch(data, block_size, batch_size):
    n = len(data) - block_size - 1
    if n <= 0:
        raise ValueError(f"Data too small ({len(data)}) for block_size {block_size}")
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, val_data, block_size, batch_size, n_eval=50):
    model.eval()
    losses = []
    for _ in range(n_eval):
        x, y = get_batch(val_data, block_size, batch_size)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    avg = sum(losses) / len(losses)
    return avg, math.exp(min(avg, 20))


def get_lr_wsd(step, n_steps, lr, warmup_frac=0.1):
    """WSD scheduler: 10% warmup, 50% stable, 40% cosine decay."""
    warmup_steps = int(n_steps * warmup_frac)
    stable_end = int(n_steps * 0.6)
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    elif step < stable_end:
        return lr
    else:
        decay_progress = (step - stable_end) / max(1, n_steps - stable_end)
        return lr * 0.5 * (1 + math.cos(math.pi * decay_progress))


def generate_sample(model, sp, start_text="def ", max_len=150, temperature=0.8):
    """Generate text sample using BPE tokenizer."""
    model.eval()
    tokens = sp.encode(start_text)
    if not tokens:
        tokens = [sp.bos_id()]
    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_len):
            logits, _, _ = model(idx[:, -model.block_size:])
            logits = logits[:, -1, :] / temperature
            # Top-p sampling (nucleus)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative prob > 0.9
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            if next_token.item() == sp.eos_id():
                break
            idx = torch.cat([idx, next_token], dim=1)

    generated = sp.decode(idx[0].tolist())
    return generated


@torch.no_grad()
def analyze_routing(model, val_data, sp, block_size, n_batches=20):
    """Analyze which experts are being used and for what content."""
    model.eval()
    expert_names = model.EXPERT_NAMES[:model.n_experts]
    expert_usage = {name: 0.0 for name in expert_names}
    total_tokens = 0

    for _ in range(n_batches):
        x, _ = get_batch(val_data, block_size, 8)
        _, _, info = model(x)
        routing = info['routing']  # (B, T, n_experts)

        for i, name in enumerate(expert_names):
            expert_usage[name] += (routing[:, :, i] > 0.1).float().sum().item()
        total_tokens += routing.shape[0] * routing.shape[1]

    print("\n  Expert Routing Analysis:")
    for name in expert_names:
        pct = 100 * expert_usage[name] / total_tokens
        bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
        print(f"    {name:8s}: {bar} {pct:.1f}%")


# ==================== Training Phases ====================

def phase0_tokenizer(args):
    """Phase 0: Train BPE tokenizer."""
    print("\n" + "=" * 70)
    print("  Phase 0: Training BPE Tokenizer")
    print("=" * 70)
    sp = train_bpe_tokenizer(vocab_size=args.vocab_size)
    print(f"  Vocab size: {sp.get_piece_size()}")
    # Show some example encodings
    examples = ["def fibonacci(n):", "Информация о системе", "import torch", "NautilusHierarchy"]
    print("  Example encodings:")
    for ex in examples:
        tokens = sp.encode(ex)
        pieces = [sp.id_to_piece(t) for t in tokens]
        print(f"    '{ex}' → {len(tokens)} tokens: {pieces[:8]}{'...' if len(pieces) > 8 else ''}")
    return sp


def phase1_train_all(model, sp, args):
    """Phase 1: Train Core + all Experts end-to-end on all data."""
    print("\n" + "=" * 70)
    print("  Phase 1: End-to-End Training (Core + All Experts)")
    print("=" * 70)

    # Collect and encode all data
    print("\n  Collecting all text data...")
    texts = collect_text_from_repos()
    train_data, val_data = encode_data(sp, texts)

    n_params = sum(p.numel() for p in model.parameters())
    n_core = sum(p.numel() for n, p in model.named_parameters() if 'expert' not in n and 'router' not in n)
    n_expert = sum(p.numel() for n, p in model.named_parameters() if 'expert' in n)
    n_router = sum(p.numel() for n, p in model.named_parameters() if 'router' in n)

    print(f"\n  Model: NautilusMoME ({n_params:,} total params)")
    print(f"    Core:    {n_core:,} params (always active)")
    print(f"    Experts: {n_expert:,} params ({n_expert//model.n_experts:,} each × {model.n_experts})")
    print(f"    Router:  {n_router:,} params")
    print(f"    Tokens/param ratio: {len(train_data)/n_params:.1f}x")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Resume?
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except Exception:
                pass
        start_step = ckpt.get('step', 0)
        print(f"  Resumed from step {start_step}")

    history = []
    t0 = time.time()
    best_val = float('inf')
    best_ppl = float('inf')

    print(f"\n  Training on {len(train_data):,} BPE tokens...")
    print(f"  {'='*60}")

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        model.set_step(step)

        cur_lr = get_lr_wsd(step, args.steps, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        x, y = get_batch(train_data, args.block_size, args.batch_size)
        logits, loss, info = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == 1:
            vl, ppl = evaluate(model, val_data, args.block_size, args.batch_size)
            if vl < best_val:
                best_val = vl
                best_ppl = ppl

            elapsed = time.time() - t0
            tokens_per_sec = (step - start_step) * args.batch_size * args.block_size / max(elapsed, 1)

            print(f"  Step {step:6d}: train={loss.item():.4f} val={vl:.4f} "
                  f"ppl={ppl:.2f} lr={cur_lr:.6f} aux={info['aux_loss']:.4f} "
                  f"[{elapsed:.0f}s, {tokens_per_sec:.0f} tok/s]")

            history.append({
                'step': step, 'train': loss.item(), 'val': vl,
                'ppl': ppl, 'lr': cur_lr, 'aux_loss': info['aux_loss'],
            })

        if step % (args.eval_every * 5) == 0:
            sample = generate_sample(model, sp, start_text="def ", max_len=100)
            print(f"  >>> Sample: {sample[:200]}")

        if step % (args.eval_every * 2) == 0:
            torch.save({
                'step': step, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val': best_val, 'best_ppl': best_ppl,
                'args': vars(args), 'phase': 1,
            }, CHECKPOINT_PATH)

    elapsed = time.time() - t0
    final_val, final_ppl = evaluate(model, val_data, args.block_size, args.batch_size, n_eval=100)

    print(f"\n  Phase 1 FINAL: val={final_val:.4f} ppl={final_ppl:.2f} "
          f"best_ppl={best_ppl:.2f} time={elapsed:.1f}s")

    # Analyze routing
    analyze_routing(model, val_data, sp, args.block_size)

    # Generate samples
    print("\n  === Generated Samples ===")
    for prompt in ["def ", "import ", "class ", "# ", "Информация"]:
        sample = generate_sample(model, sp, start_text=prompt, max_len=120)
        print(f"  [{prompt}]: {sample[:250]}")
        print()

    return history, final_val, final_ppl, best_val, best_ppl, elapsed, train_data, val_data


def phase2_expert_finetune(model, sp, args):
    """Phase 2: Fine-tune individual experts on domain-specific data."""
    print("\n" + "=" * 70)
    print("  Phase 2: Expert Fine-Tuning (Domain-Specific)")
    print("=" * 70)

    # Prepare domain data (skip if already cached)
    domain_files_exist = all(
        os.path.exists(os.path.join(DOMAIN_DIR, f'{d}.txt'))
        for d in EXPERT_DOMAINS
    )
    if not domain_files_exist:
        print("\n  Preparing domain-specific data...")
        prepare_domain_data()
    else:
        print("\n  Using cached domain data from", DOMAIN_DIR)
    domain_data = encode_domain_data(sp)

    # Freeze core, train each expert on its domain
    model.freeze_core()

    expert_results = {}
    for expert_name in model.EXPERT_NAMES[:model.n_experts]:
        if expert_name not in domain_data:
            print(f"\n  SKIP {expert_name}: no domain data")
            continue

        dd = domain_data[expert_name]
        if len(dd['train']) < args.block_size * 2:
            print(f"\n  SKIP {expert_name}: insufficient data ({len(dd['train'])} tokens)")
            continue

        print(f"\n  Fine-tuning Expert: {expert_name} "
              f"({EXPERT_DOMAINS[expert_name]['name']})")
        print(f"    Data: {len(dd['train']):,} train tokens")

        # Freeze all experts except current
        model.freeze_all_experts_except(expert_name)
        # Also unfreeze bridge and router for adaptation
        for p in model.bridge.parameters():
            p.requires_grad = True
        for p in model.router.parameters():
            p.requires_grad = True

        expert_lr = args.lr * 0.3  # Lower LR for fine-tuning
        expert_steps = min(args.expert_steps, len(dd['train']) // (args.batch_size * args.block_size) * 3)
        expert_steps = max(expert_steps, 200)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=expert_lr, weight_decay=args.wd,
        )

        best_expert_val = float('inf')
        for step in range(1, expert_steps + 1):
            model.train()
            cur_lr = get_lr_wsd(step, expert_steps, expert_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = cur_lr

            x, y = get_batch(dd['train'], args.block_size, args.batch_size)
            _, loss, info = model(x, targets=y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % max(expert_steps // 5, 1) == 0 or step == 1:
                vl, ppl = evaluate(model, dd['val'], args.block_size, args.batch_size, n_eval=20)
                if vl < best_expert_val:
                    best_expert_val = vl
                print(f"    Step {step:4d}/{expert_steps}: "
                      f"train={loss.item():.4f} val={vl:.4f} ppl={ppl:.2f}")

        expert_results[expert_name] = {
            'steps': expert_steps,
            'best_val': best_expert_val,
            'train_tokens': len(dd['train']),
        }

    # Unfreeze all for potential further training
    model.unfreeze_all()

    print("\n  Expert Fine-Tuning Summary:")
    for name, res in expert_results.items():
        print(f"    {name:8s}: {res['steps']} steps, "
              f"best_val={res['best_val']:.4f}, "
              f"data={res['train_tokens']:,} tokens")

    return expert_results


def phase3_router_finetune(model, sp, train_data, val_data, args):
    """Phase 3: Fine-tune router on mixed data (all experts active)."""
    print("\n" + "=" * 70)
    print("  Phase 3: Router Fine-Tuning")
    print("=" * 70)

    # Only train router parameters
    for p in model.parameters():
        p.requires_grad = False
    for p in model.router.parameters():
        p.requires_grad = True
    for p in model.bridge.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 0.1, weight_decay=args.wd,
    )

    router_steps = args.router_steps
    print(f"  Training router for {router_steps} steps...")

    for step in range(1, router_steps + 1):
        model.train()
        cur_lr = get_lr_wsd(step, router_steps, args.lr * 0.1)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        x, y = get_batch(train_data, args.block_size, args.batch_size)
        _, loss, info = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % max(router_steps // 5, 1) == 0:
            vl, ppl = evaluate(model, val_data, args.block_size, args.batch_size, n_eval=30)
            print(f"    Step {step:4d}/{router_steps}: val={vl:.4f} ppl={ppl:.2f} "
                  f"aux={info['aux_loss']:.4f}")

    model.unfreeze_all()

    # Final routing analysis
    analyze_routing(model, val_data, sp, args.block_size)


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Train NautilusMoME')
    # Model
    parser.add_argument('--vocab-size', type=int, default=4096, help='BPE vocab size')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=4, help='Core transformer layers')
    parser.add_argument('--n-heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--block-size', type=int, default=256, help='Context window (BPE tokens, ~768 bytes)')
    parser.add_argument('--d-expert', type=int, default=128, help='Expert hidden dimension')
    parser.add_argument('--n-experts', type=int, default=6, help='Number of micro-experts')
    parser.add_argument('--top-k', type=int, default=2, help='Top-k experts per token')
    # Training
    parser.add_argument('--steps', type=int, default=5000, help='Phase 1 training steps')
    parser.add_argument('--expert-steps', type=int, default=1000, help='Phase 2 steps per expert')
    parser.add_argument('--router-steps', type=int, default=500, help='Phase 3 router steps')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--eval-every', type=int, default=500, help='Eval interval')
    # Control
    parser.add_argument('--phase', type=int, default=0, help='Start from phase (0/1/2/3)')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume')
    args = parser.parse_args()

    set_seed(42)

    # Phase 0: BPE Tokenizer
    if args.phase <= 0:
        sp = phase0_tokenizer(args)
    else:
        sp = load_tokenizer()

    vocab_size = sp.get_piece_size()

    # Create model
    model = NautilusMoME(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        block_size=args.block_size,
        d_expert=args.d_expert,
        n_experts=args.n_experts,
        top_k=args.top_k,
    )

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        print(f"  Loaded checkpoint: {args.resume}")

    # Phase 1: End-to-end training
    train_data = val_data = None
    if args.phase <= 1:
        result = phase1_train_all(model, sp, args)
        history, final_val, final_ppl, best_val, best_ppl, elapsed, train_data, val_data = result
    else:
        # Load data for later phases
        texts = collect_text_from_repos()
        train_data, val_data = encode_data(sp, texts)
        history = []
        final_val = best_val = float('inf')
        final_ppl = best_ppl = float('inf')
        elapsed = 0

    # Phase 2: Expert fine-tuning
    if args.phase <= 2:
        expert_results = phase2_expert_finetune(model, sp, args)
    else:
        expert_results = {}

    # Phase 3: Router fine-tuning
    if args.phase <= 3 and train_data is not None:
        phase3_router_finetune(model, sp, train_data, val_data, args)

    # Final evaluation
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION (all phases complete)")
    print("=" * 70)

    if val_data is not None:
        final_val, final_ppl = evaluate(model, val_data, args.block_size, args.batch_size, n_eval=100)
        print(f"  Final val_loss={final_val:.4f}, PPL={final_ppl:.2f}")

    # Final samples
    print("\n  === Final Generated Samples ===")
    prompts = [
        "def fibonacci(",
        "import os\nimport ",
        "class NautilusModel(",
        "# Mathematical formula for",
        "Информация о ",
        "function handleClick(",
    ]
    for prompt in prompts:
        sample = generate_sample(model, sp, start_text=prompt, max_len=150, temperature=0.7)
        print(f"  [{prompt[:30]}]: {sample[:300]}")
        print()

    # Final routing analysis
    if val_data is not None:
        analyze_routing(model, val_data, sp, args.block_size)

    # Save results
    n_params = sum(p.numel() for p in model.parameters())
    results = {
        'model': 'NautilusMoME',
        'architecture': {
            'vocab_size': vocab_size,
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'block_size': args.block_size,
            'd_expert': args.d_expert,
            'n_experts': args.n_experts,
            'top_k': args.top_k,
            'tokenizer': 'BPE (sentencepiece)',
        },
        'params': {
            'total': n_params,
            'core': sum(p.numel() for n, p in model.named_parameters()
                       if 'expert' not in n and 'router' not in n and 'bridge' not in n),
            'experts': sum(p.numel() for n, p in model.named_parameters() if 'expert' in n),
            'router': sum(p.numel() for n, p in model.named_parameters() if 'router' in n),
            'bridge': sum(p.numel() for n, p in model.named_parameters() if 'bridge' in n),
        },
        'expert_domains': {k: v['name'] for k, v in EXPERT_DOMAINS.items()},
        'training': {
            'optimizer': 'adamw',
            'lr': args.lr,
            'weight_decay': args.wd,
            'scheduler': 'wsd',
            'phase1_steps': args.steps,
            'phase2_expert_steps': args.expert_steps,
            'phase3_router_steps': args.router_steps,
        },
        'final_val': final_val,
        'final_ppl': final_ppl,
        'best_val': best_val if best_val != float('inf') else None,
        'best_ppl': best_ppl if best_ppl != float('inf') else None,
        'time_seconds': elapsed,
        'expert_results': expert_results,
        'history': history,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  >> Saved results to {RESULTS_PATH}")

    # Save final checkpoint
    torch.save({
        'step': args.steps,
        'model': model.state_dict(),
        'best_val': best_val,
        'best_ppl': best_ppl,
        'args': vars(args),
        'phase': 3,
    }, CHECKPOINT_PATH)
    print(f"  >> Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == '__main__':
    main()

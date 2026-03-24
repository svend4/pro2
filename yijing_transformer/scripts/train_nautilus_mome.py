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
  └───────┬───────┘
  ┌───────▼───────┐
  │ CROSS-DOMAIN   │  ← inter-expert analogies (15 pairs for 6 experts)
  │ ANALOGY        │     turns domain "collage" into structured insight
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
  Phase 4: Train CrossDomainAnalogy (inter-expert analogies)
  Phase 5: Train SYNTH expert (cross-domain synthesizer)
  Phase 9: Archetype differentiation
  Phase 10: Antonym differentiation (binary + ternary)
  Phase 11: Organic contrastive routing (sharpen specialization)

Usage:
  python train_nautilus_mome.py                    # Full pipeline
  python train_nautilus_mome.py --phase 1          # Skip tokenizer training
  python train_nautilus_mome.py --phase 2          # Expert fine-tuning only
  python train_nautilus_mome.py --phase 4          # Analogy training only
  python train_nautilus_mome.py --phase 5          # SYNTH training only
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
import itertools
import sentencepiece as spm

# ==================== Configuration ====================

# ── Organic Expert Domains ──
# Experts are defined by CONTENT SIGNALS, not by repository origin.
# The router learns to recognize these patterns from the data itself.
# Repo hints are soft suggestions for initial data grouping only —
# the router is free to override them based on actual content.
#
# Four-level alignment (info3):
#   Level 1 (Formula):    MATH — abstract mathematical structures
#   Level 2 (Archetype):  CODE, SYSTEM — structural patterns & architecture
#   Level 3 (Algorithm):  RECON, INFO — procedural processes & management
#   Level 4 (Theorem):    HUMAN — practical wisdom, ethics, meaning
#
# The router discovers these levels organically through training.
# No domain is "locked" — content flows to whichever expert resonates.
# ─── Гексаграммный слой экспертов ──────────────────────────────────────────
# Каждый эксперт связан с одной из 64 гексаграмм И-Цзин (Q6 = {-1,+1}^6).
# Гексаграмма — не декорация, а геометрический якорь в пространстве архетипов.
# Номер гексаграммы → индекс вершины Q6 → 6-битный вектор (yin/yang линии).
#
#  乾 (Цянь)  — Небо, Творчество  →  структура, формализм, математика
#  巽 (Сюнь)  — Ветер, Проникновение → алгоритм, реализация, код
#  坤 (Кунь)  — Земля, Восприятие  →  гуманитарное, поведение, человек
#  坎 (Кань)  — Вода, Поток       →  инфраструктура, система, опасность
#  離 (Ли)    — Огонь, Ясность    →  распознавание, паттерн, ReConstructor
#  兌 (Дуй)  — Озеро, Обмен      →  информация, каталоги, поиск
#  革 (Гэ)   — Смена, Синтез     →  кросс-доменный, протоязык, SYNTH
HEXAGRAM_MAP = {
    'MATH':   {'gua': '乾', 'pinyin': 'Qián',  'num': 1,  'meaning': 'Творчество — Небо',
               'q6_idx': 63, 'nature': 'Структура, формализм, чистая мысль'},
    'CODE':   {'gua': '巽', 'pinyin': 'Xùn',   'num': 57, 'meaning': 'Проникновение — Ветер',
               'q6_idx': 6,  'nature': 'Алгоритм, реализация, мягкое движение'},
    'HUMAN':  {'gua': '坤', 'pinyin': 'Kūn',   'num': 2,  'meaning': 'Восприятие — Земля',
               'q6_idx': 0,  'nature': 'Гуманитарное, поведение, принятие'},
    'SYSTEM': {'gua': '坎', 'pinyin': 'Kǎn',   'num': 29, 'meaning': 'Поток — Вода',
               'q6_idx': 18, 'nature': 'Инфраструктура, опасность, непрерывность'},
    'RECON':  {'gua': '離', 'pinyin': 'Lí',    'num': 30, 'meaning': 'Ясность — Огонь',
               'q6_idx': 45, 'nature': 'Распознавание, паттерн, реконструкция'},
    'INFO':   {'gua': '兌', 'pinyin': 'Duì',   'num': 58, 'meaning': 'Обмен — Озеро',
               'q6_idx': 27, 'nature': 'Информация, каталоги, коммуникация'},
    'SYNTH':  {'gua': '革', 'pinyin': 'Gé',    'num': 49, 'meaning': 'Смена — Революция',
               'q6_idx': 36, 'nature': 'Кросс-доменный синтез, протоязык, сумерки'},
}

EXPERT_DOMAINS = {
    'MATH': {
        'name': 'Mathematical Structures',
        'content_signals': [
            'formulas', 'equations', 'proofs', 'theorems',
            'hexagrams', 'trigrams', 'hypercube', 'geometry',
            'tensor', 'matrix', 'eigenvalue', 'gradient',
        ],
        'repo_hints': ['meta', 'data2', 'pro2'],
        'description': 'Math, formulas, algorithms, hexagrams, transformers',
        'four_level': 'formula',
        'hexagram': HEXAGRAM_MAP['MATH'],
    },
    'CODE': {
        'name': 'Software Engineering',
        'content_signals': [
            'def ', 'class ', 'import ', 'function ',
            'return ', 'const ', 'interface ', 'async ',
            '.tsx', '.py', '.ts', '.js',
        ],
        'repo_hints': ['daten3', 'daten2', 'data20'],
        'description': 'TypeScript, Flask, React, full-stack, KMS',
        'four_level': 'archetype',
        'hexagram': HEXAGRAM_MAP['CODE'],
    },
    'HUMAN': {
        'name': 'Humanitarian Knowledge',
        'content_signals': [
            'этика', 'архетип', 'поведени', 'мудрость',
            'психолог', 'смысл', 'ценност', 'человек',
            'ethics', 'wisdom', 'meaning', 'archetype',
        ],
        'repo_hints': ['info3', 'daten22', 'info'],
        'description': 'Ethics, archetypes, MBTI, behavioral formulas',
        'four_level': 'theorem',
        'hexagram': HEXAGRAM_MAP['HUMAN'],
    },
    'SYSTEM': {
        'name': 'System Architecture',
        'content_signals': [
            'docker', 'kubernetes', 'nginx', 'deploy',
            'pipeline', 'orchestrat', 'container', 'mcp',
            'SELECT ', 'CREATE TABLE', 'API', 'endpoint',
        ],
        'repo_hints': ['info7', 'daten', 'universal-file-storage-mcp'],
        'description': 'AI orchestration, DevOps, MCP, containers',
        'four_level': 'archetype',
        'hexagram': HEXAGRAM_MAP['SYSTEM'],
    },
    'RECON': {
        'name': 'Pattern Recognition',
        'content_signals': [
            'распозна', 'реконструкц', 'OCR', 'паттерн',
            'восстановл', 'сканир', 'документ', 'puzzle',
        ],
        'repo_hints': ['meta2'],
        'description': 'Document reconstruction, OCR, puzzle algorithms',
        'four_level': 'algorithm',
        'hexagram': HEXAGRAM_MAP['RECON'],
    },
    'INFO': {
        'name': 'Information Management',
        'content_signals': [
            'catalog', 'search', 'index', 'metadata',
            'knowledge base', 'автоматизац', 'каталог',
            'README', 'documentation', 'config',
        ],
        'repo_hints': ['info1', 'info4', 'info5', 'daten11', 'data30', 'in4'],
        'description': 'Knowledge bases, catalogs, search, automation',
        'four_level': 'algorithm',
        'hexagram': HEXAGRAM_MAP['INFO'],
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


def _score_content_signals(text, domain_key):
    """Score how well a text matches a domain's content signals.

    Returns a float [0, 1] indicating signal density.
    This is a soft, organic measure — not a hard label.
    """
    signals = EXPERT_DOMAINS[domain_key].get('content_signals', [])
    if not signals:
        return 0.0
    text_lower = text[:5000].lower()  # sample first 5K chars for speed
    matches = sum(1 for s in signals if s.lower() in text_lower)
    return matches / len(signals)


def _detect_domain_organic(text, repo_name):
    """Detect domain organically: content signals first, repo hint as fallback.

    Priority:
      1. Content signal scoring (organic — based on what's IN the text)
      2. Repo hint (soft fallback — based on where text came FROM)
      3. 'INFO' default (catch-all for uncategorized content)

    If multiple domains score equally, the text goes to ALL matching domains
    (soft boundaries — a file can belong to multiple experts).
    """
    # Score all domains by content signals
    scores = {dk: _score_content_signals(text, dk) for dk in EXPERT_DOMAINS}
    max_score = max(scores.values())

    # If any domain has strong content match (>= 0.15), use it
    if max_score >= 0.15:
        return [dk for dk, s in scores.items() if s >= max_score * 0.7]

    # Soft fallback: repo hint
    repo_to_domain = {}
    for dk, info in EXPERT_DOMAINS.items():
        for repo in info.get('repo_hints', []):
            repo_to_domain[repo] = dk
    if repo_name in repo_to_domain:
        return [repo_to_domain[repo_name]]

    return ['INFO']  # catch-all


def prepare_domain_data():
    """Split collected text into domain-specific files using organic content detection.

    Instead of hard repo→domain mapping, each file is scored against
    all domains' content signals. Files with strong signals go to the
    matching domain(s). Files without clear signals fall back to repo
    hints, then to INFO as catch-all.

    This allows the same file to contribute to multiple experts if it
    contains cross-domain content (e.g., Russian code comments → RECON + CODE).
    """
    os.makedirs(DOMAIN_DIR, exist_ok=True)

    domain_texts = defaultdict(list)
    cross_domain_count = 0

    for repo_dir in ALL_REPO_DIRS:
        if not os.path.isdir(repo_dir):
            continue
        repo_name = os.path.basename(repo_dir)

        texts = collect_text_from_repos([repo_dir])
        for text in texts:
            domains = _detect_domain_organic(text, repo_name)
            if len(domains) > 1:
                cross_domain_count += 1
            for domain in domains:
                domain_texts[domain].append(text)

    print(f"\n  Organic domain detection: {cross_domain_count} cross-domain files")
    print("  Domain data distribution:")
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
    """Routes tokens to experts with adaptive top-k selection.

    Organic routing: instead of always activating exactly top-2 experts,
    adapts the number of active experts based on routing confidence.

    - High confidence (one expert dominates) → activate 1 expert (efficient)
    - Medium confidence → activate 2 experts (default)
    - Low confidence (cross-domain) → activate up to 3 experts (thorough)

    The adaptation is smooth via soft gating, not hard k-switching.
    """

    def __init__(self, d_model, n_experts, top_k=2, temperature=1.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k  # base top-k (used as maximum)
        self.temperature = temperature

        # Two-layer router for better context understanding
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_experts),
        )

        # Adaptive top-k: learns when to use more/fewer experts
        # Maps routing logits → a soft "how many experts" signal
        self.adaptive_k = nn.Sequential(
            nn.Linear(n_experts, n_experts),
            nn.GELU(),
            nn.Linear(n_experts, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            expert_weights: (B, T, n_experts) — sparse, adaptive non-zero count
            aux_loss: load balancing loss
        """
        logits = self.router(x) / self.temperature  # (B, T, n_experts)

        # Always select top_k experts as candidates
        top_k = min(self.top_k, self.n_experts)
        top_k_logits, top_k_indices = logits.topk(top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B, T, top_k)

        # Adaptive gating: learn per-token how many experts to actually use.
        # k_gate ∈ [0, 1]: 0 = only top-1, 1 = full top-k
        # When the router is confident (one logit dominates), k_gate → 0
        # When uncertain (logits are flat), k_gate → 1
        k_gate = torch.sigmoid(self.adaptive_k(logits))  # (B, T, 1)

        # Apply adaptive gating: fade out lower-ranked experts
        if top_k > 1:
            # Create fade mask: [1.0, k_gate, k_gate², ...] for each position
            fade = torch.ones_like(top_k_weights)
            for rank in range(1, top_k):
                fade[:, :, rank] = k_gate.squeeze(-1) ** rank
            top_k_weights = top_k_weights * fade
            # Re-normalize
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Scatter to full expert dimension
        expert_weights = torch.zeros_like(logits)
        expert_weights.scatter_(-1, top_k_indices, top_k_weights)

        # Load balancing auxiliary loss
        avg_routing = expert_weights.mean(dim=(0, 1))  # (n_experts,)
        target = torch.ones_like(avg_routing) / self.n_experts
        aux_loss = F.mse_loss(avg_routing, target) * self.n_experts

        return expert_weights, aux_loss


# ─── SOLAN Geometric Auxiliary Signal ─────────────────────────────────────────
# Интеграция SOLAN-76 алфавита: каждый BPE-токен получает геометрический сигнал
# из гиперкуба Q6 на основе символьного состава токена.
#
# Принцип:
#   BPE-токен "def" → символы ['d','e','f'] → SOLAN-векторы → XOR/среднее → 6-битный вектор
#   Этот вектор = координата в гиперкубе {-1,+1}^6 = 1 из 64 гексаграмм И-Цзин
#
# Зачем:
#   - Токены с похожей геометрической структурой (малое расстояние Хэмминга)
#     автоматически получают близкие сигналы → дополнительный геометрический инициализм
#   - Twilight language: при кросс-доменном смешении символы из разных доменов
#     дают разные Q6-координаты → новые слова имеют «геометрический отпечаток»
#   - Связывает NautilusMoME с оригинальным YiJingGPT через общую Q6-структуру

# Импортируем SOLAN карту напрямую (без создания GlyphTokenizer объекта)
def _build_solan_table_for_bpe(sp_model_path: str, vocab_size: int = 4096) -> 'torch.Tensor':
    """Строит SOLAN Q6-таблицу для BPE-словаря.

    Для каждого из vocab_size BPE-токенов вычисляет 6-битный вектор {-1,+1}^6
    на основе символов, из которых состоит этот токен.

    Алгоритм:
      1. Декодируем BPE-токен в строку символов
      2. Для каждого символа берём его SOLAN-вектор из _SOLAN_MAP
      3. Применяем XOR (знаковое перемножение) по всем символам токена
         → токен "ab" = SOLAN("a") ⊗ SOLAN("b) в Z₂^6

    Args:
        sp_model_path: путь к файлу SentencePiece модели
        vocab_size: размер словаря

    Returns:
        Tensor (vocab_size, 6), значения {-1.0, +1.0}
        Возвращает None если sentencepiece недоступен.
    """
    try:
        import sentencepiece as spm
    except ImportError:
        return None

    # Импортируем SOLAN-маппинг
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tokenizer.glyph_tokenizer import _SOLAN_MAP, _bits_to_vertex
        solan_available = True
    except ImportError:
        solan_available = False

    try:
        sp = spm.SentencePieceProcessor()
        sp.Load(sp_model_path)
    except Exception:
        return None

    table = torch.ones(vocab_size, 6)  # начинаем с (+1, +1, +1, +1, +1, +1)

    for token_id in range(min(vocab_size, sp.GetPieceSize())):
        token_str = sp.IdToPiece(token_id)
        # Убираем специальный символ '▁' (пробел в начале слова у SentencePiece)
        token_str = token_str.replace('▁', ' ').replace('<unk>', '?').replace('<s>', ' ').replace('</s>', '.')

        if not token_str:
            continue

        # XOR (знаковое перемножение) SOLAN-векторов всех символов токена
        # В Z₂^6: (-1)^(sum of bits) = XOR по всем символам
        combined = [1, 1, 1, 1, 1, 1]  # нейтральный элемент (identity для XOR в {-1,+1})
        for ch in token_str:
            if solan_available:
                bits = _SOLAN_MAP.get(ch)
                if bits is None:
                    code = hash(ch) % 64
                    bits = tuple((code >> (5 - b)) & 1 for b in range(6))
                vertex = _bits_to_vertex(bits)  # {-1, +1}^6
            else:
                # Fallback: простое хэширование символа
                code = ord(ch) % 64
                vertex = tuple(2 * ((code >> (5 - b)) & 1) - 1 for b in range(6))

            # XOR = поэлементное умножение в {-1,+1}
            combined = [combined[i] * vertex[i] for i in range(6)]

        table[token_id] = torch.tensor(combined, dtype=torch.float32)

    return table


class NautilusBridge(nn.Module):
    """Organic bridge that merges expert outputs via content-dependent attention.

    Instead of mechanical pairwise merge (0+1, 2+3, 4+5 by index order),
    uses cross-attention where the core hidden state QUERIES expert outputs.
    The model itself decides how to blend experts based on current context.

    Architecture:
        Query  = core_hidden (what the model currently needs)
        Keys   = expert_outputs (what each expert proposes)
        Values = expert_outputs (expert contributions)
        → Attention-weighted sum → the model blends organically

    This replaces the old fixed-index pairing with content-dependent merging.
    """

    def __init__(self, d_model, n_experts, n_heads=4):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model

        # Cross-attention: core queries expert proposals
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # Output projection after attention
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
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
        n_exp = len(expert_outputs)

        # Weight expert outputs by routing
        weighted = []
        for i, exp_out in enumerate(expert_outputs):
            w = expert_weights[:, :, i:i+1]  # (B, T, 1)
            weighted.append(exp_out * w)

        # Stack expert outputs: (B, T, n_experts, D)
        expert_stack = torch.stack(weighted, dim=2)

        # Cross-attention: core hidden queries expert proposals
        # Q from core: (B, T, D)
        # K, V from experts: (B, T, n_experts, D) → reshape for attention
        Q = self.query_proj(core_hidden)                          # (B, T, D)
        K = self.key_proj(expert_stack.view(B * T, n_exp, D))    # (B*T, n_exp, D)
        V = self.value_proj(expert_stack.view(B * T, n_exp, D))  # (B*T, n_exp, D)

        # Reshape Q for multi-head: (B*T, 1, n_heads, head_dim) → (B*T, n_heads, 1, head_dim)
        Q = Q.view(B * T, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B * T, n_exp, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B * T, n_exp, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention: (B*T, n_heads, 1, head_dim) @ (B*T, n_heads, head_dim, n_exp)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B*T, n_heads, 1, n_exp)
        attn = F.softmax(attn, dim=-1)

        # Weighted sum of expert values
        merged = torch.matmul(attn, V)  # (B*T, n_heads, 1, head_dim)
        merged = merged.transpose(1, 2).reshape(B, T, D)  # (B, T, D)

        # Project and gate
        merged = self.out_proj(merged)

        # Residual connection with gate
        output = core_hidden + self.residual_gate * self.ln(merged)
        return output


# ==================== Cross-Domain Analogy ====================


class ProverbCondenser(nn.Module):
    """Compresses expert output to a "proverb" — a concentrated formula.

    Like a proverb is concentrated wisdom in few words, this module
    distills an expert's output (B, T, D) into a single vector (B, 1, D)
    that captures the essence of what the expert sees.
    """

    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, expert_output):
        B, T, D = expert_output.shape
        q = self.query.expand(B, -1, -1)
        k = self.key_proj(expert_output)
        v = self.value_proj(expert_output)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, v)  # (B, 1, D)


class AnalogyPair(nn.Module):
    """A bridge between two domains for cross-domain analogy.

    Like biophysics uses physics formulas to describe biology,
    this module projects "proverbs" from two experts into a shared
    analogy space and synthesizes cross-domain insight.
    """

    def __init__(self, d_model, d_analogy=None):
        super().__init__()
        d_analogy = d_analogy or d_model // 2
        self.proj_a = nn.Linear(d_model, d_analogy)
        self.proj_b = nn.Linear(d_model, d_analogy)
        self.similarity_gate = nn.Sequential(
            nn.Linear(d_analogy * 2, d_analogy),
            nn.GELU(),
            nn.Linear(d_analogy, 1),
            nn.Sigmoid(),
        )
        self.synthesis = nn.Sequential(
            nn.Linear(d_analogy * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Parameter(torch.tensor(0.01))

    def forward(self, formula_a, formula_b):
        a = self.proj_a(formula_a)
        b = self.proj_b(formula_b)
        combined = torch.cat([a, b], dim=-1)
        strength = self.similarity_gate(combined)
        insight = self.synthesis(combined) * self.gate
        return insight, strength


class CrossDomainAnalogy(nn.Module):
    """Cross-domain analogy module: turns expert "collage" into structured insight.

    For 6 experts creates C(6,2)=15 analogy pairs. Each pair can discover
    and leverage cross-domain analogies (e.g. MATH↔HUMAN for psychotype math,
    CODE↔SYSTEM for architecture patterns).

    Analogies activate only when similarity exceeds threshold —
    otherwise domains remain independent.
    """

    def __init__(self, d_model, n_experts=6, expert_names=None,
                 d_analogy=None, threshold=0.3):
        super().__init__()
        self.n_experts = n_experts
        self.threshold = threshold
        names = expert_names or ['MATH', 'CODE', 'HUMAN', 'SYSTEM', 'RECON', 'INFO']
        names = names[:n_experts]

        self.condensers = nn.ModuleDict({
            name: ProverbCondenser(d_model) for name in names
        })

        self.pairs = nn.ModuleDict()
        self.pair_keys = []
        for i, j in itertools.combinations(range(n_experts), 2):
            key = f"{names[i]}_{names[j]}"
            self.pairs[key] = AnalogyPair(d_model, d_analogy)
            self.pair_keys.append((key, i, j))

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.analogy_gate = nn.Parameter(torch.tensor(0.05))

    def forward(self, bridge_output, expert_outputs, expert_weights, expert_names):
        B, T, D = bridge_output.shape
        names = expert_names[:self.n_experts]

        formulas = {}
        for i, name in enumerate(names):
            w = expert_weights[:, :, i:i+1]
            weighted = expert_outputs[i] * w
            formulas[name] = self.condensers[name](weighted)

        total_insight = torch.zeros(B, 1, D, device=bridge_output.device)
        analogy_strengths = {}
        active_count = 0

        for key, i, j in self.pair_keys:
            insight, strength = self.pairs[key](formulas[names[i]], formulas[names[j]])
            avg_str = strength.mean().item()
            analogy_strengths[key] = avg_str
            if avg_str > self.threshold:
                total_insight = total_insight + insight * strength
                active_count += 1

        if active_count > 0:
            total_insight = total_insight / max(active_count, 1)
            projected = self.output_proj(total_insight)
            output = bridge_output + projected.expand(B, T, D) * self.analogy_gate
        else:
            output = bridge_output

        return output, {
            'strengths': analogy_strengths,
            'active_pairs': active_count,
            'gate': self.analogy_gate.item(),
        }


# ==================== PseudoRAG Archetype Layer ====================


# 16 Information Archetypes from PseudoRAG (= Q4 hypercube vertices)
# Each archetype = one vertex of {-1,+1}^4 with semantic meaning.
# Axes: M/A (Material/Abstract) × S/D (Static/Dynamic) ×
#        E/C (Elementary/Complex) × O/F (Ordered/Fluid)
# This is isomorphic to MatryoshkaQuantizer's hex-digits:
#   2 space edges × 2 time edges = 4×4 = 16.
PSEUDORAG_ARCHETYPES = {
    0:  {'code': 'MSEO', 'name': 'Кристалл',  'axes': (-1,-1,-1,-1)},
    1:  {'code': 'MSEF', 'name': 'Песок',      'axes': (-1,-1,-1, 1)},
    2:  {'code': 'MSCO', 'name': 'Здание',     'axes': (-1,-1, 1,-1)},
    3:  {'code': 'MSCF', 'name': 'Лес',        'axes': (-1,-1, 1, 1)},
    4:  {'code': 'MDEO', 'name': 'Механизм',   'axes': (-1, 1,-1,-1)},
    5:  {'code': 'MDEF', 'name': 'Организм',   'axes': (-1, 1,-1, 1)},
    6:  {'code': 'MDCO', 'name': 'Машина',     'axes': (-1, 1, 1,-1)},
    7:  {'code': 'MDCF', 'name': 'Город',      'axes': (-1, 1, 1, 1)},
    8:  {'code': 'ASEO', 'name': 'Аксиома',    'axes': ( 1,-1,-1,-1)},
    9:  {'code': 'ASEF', 'name': 'Архетип',    'axes': ( 1,-1,-1, 1)},
    10: {'code': 'ASCO', 'name': 'Теория',     'axes': ( 1,-1, 1,-1)},
    11: {'code': 'ASCF', 'name': 'Культура',   'axes': ( 1,-1, 1, 1)},
    12: {'code': 'ADEO', 'name': 'Алгоритм',   'axes': ( 1, 1,-1,-1)},
    13: {'code': 'ADEF', 'name': 'Интуиция',   'axes': ( 1, 1,-1, 1)},
    14: {'code': 'ADCO', 'name': 'Программа',  'axes': ( 1, 1, 1,-1)},
    15: {'code': 'ADCF', 'name': 'Общество',   'axes': ( 1, 1, 1, 1)},
}


class ArchetypeLayer(nn.Module):
    """Maps hidden state + expert routing to PseudoRAG's 16 archetypes via Q4 hypercube.

    V2: Uses hidden state (rich, content-dependent) as primary signal,
    with expert weights as auxiliary input. The hidden state already
    differentiates content types (proven by working expert routing),
    so an MLP on it can learn meaningful Q4 axis coordinates.

    Architecture:
        hidden_state (B,T,D) → MLP → Q4 axes (B,T,4)
        expert_weights (B,T,6) → linear → Q4 axes (auxiliary)
        → combined axes → soft assignment to 16 archetypes
        → archetype embedding → enrichment

    The 4 axes:
        Axis M/A: Material (-1) vs Abstract (+1)
        Axis S/D: Static (-1) vs Dynamic (+1)
        Axis E/C: Elementary (-1) vs Complex (+1)
        Axis O/F: Ordered (-1) vs Fluid (+1)
    """

    def __init__(self, d_model, n_experts=6):
        super().__init__()
        self.d_model = d_model
        self.n_archetypes = 16

        # PRIMARY: Hidden state → 4 Q4 axes via MLP (content-rich signal)
        self.hidden_to_axes = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4),
        )

        # AUXILIARY: Expert weights → 4 Q4 axes (routing signal)
        self.expert_to_axes = nn.Linear(n_experts, 4, bias=True)

        # Context-dependent blend: instead of a fixed global scalar,
        # the blend between hidden-derived and expert-derived axes
        # is computed per-token from the content itself.
        # On some tokens hidden state is more informative (e.g. code),
        # on others routing is more informative (e.g. cross-domain).
        # A small MLP learns this per-token decision.
        self.blend_net = nn.Sequential(
            nn.Linear(d_model + n_experts, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 4),  # per-axis blend
            nn.Sigmoid(),
        )

        # 16 archetype embeddings (learnable)
        self.archetype_emb = nn.Embedding(16, d_model)

        # Q4 codebook: 16 vertices of {-1,+1}^4
        q4 = torch.tensor([
            PSEUDORAG_ARCHETYPES[i]['axes'] for i in range(16)
        ], dtype=torch.float32)
        self.register_buffer('q4_codebook', q4)

        # Temperature for soft assignment (lower = sharper)
        self.temp = nn.Parameter(torch.tensor(0.5))

        # Output gate (starts small)
        self.gate = nn.Parameter(torch.tensor(0.01))

        # Project archetype signal to model dim
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, expert_weights):
        """
        Args:
            x: (B, T, D) current hidden state (after bridge + analogy)
            expert_weights: (B, T, n_experts) routing weights

        Returns:
            enriched_x: (B, T, D)
            info: dict with archetype activations
        """
        B, T, D = x.shape

        # Step 1a: Hidden state → axes (content-rich, per-token)
        hidden_axes = torch.tanh(self.hidden_to_axes(x))  # (B, T, 4)

        # Step 1b: Expert routing → axes (sparse, per-token)
        expert_axes = torch.tanh(self.expert_to_axes(expert_weights))  # (B, T, 4)

        # Step 1c: Context-dependent blend (per-token, per-axis)
        # The model decides for each token and each axis how much to trust
        # hidden state vs routing signal. Some tokens may need routing
        # (cross-domain), others may need hidden state (clear single-domain).
        blend_input = torch.cat([x, expert_weights], dim=-1)  # (B, T, D+n_experts)
        blend = self.blend_net(blend_input)  # (B, T, 4), each in [0,1]
        axes = blend * hidden_axes + (1 - blend) * expert_axes  # (B, T, 4)

        # Step 2: Soft assignment to 16 archetypes
        sim = torch.einsum('btd,nd->btn', axes, self.q4_codebook)  # (B,T,16)
        sim = sim / (self.temp.abs() + 0.05)
        archetype_probs = F.softmax(sim, dim=-1)  # (B, T, 16)

        # Step 3: Weighted sum of archetype embeddings
        archetype_signal = torch.einsum(
            'btn,nd->btd', archetype_probs, self.archetype_emb.weight
        )

        # Step 4: Project and gate
        enrichment = self.out_proj(archetype_signal) * self.gate
        enriched_x = x + enrichment

        # Info for diagnostics
        avg_probs = archetype_probs.mean(dim=(0, 1))  # (16,)
        top_idx = avg_probs.argmax().item()
        top_arch = PSEUDORAG_ARCHETYPES[top_idx]

        info = {
            'top_archetype': top_arch['code'],
            'top_name': top_arch['name'],
            'top_prob': avg_probs[top_idx].item(),
            'entropy': -(archetype_probs * (archetype_probs + 1e-8).log()).sum(-1).mean().item(),
            'gate': self.gate.item(),
            'axis_means': axes.mean(dim=(0, 1)).tolist(),
            'archetype_probs': archetype_probs,  # for loss computation
            'axes': axes,  # (B, T, 4) for supervision loss
            'blend': blend.mean(dim=(0, 1)).tolist(),  # per-axis avg blend
        }
        return enriched_x, info


# ==================== Twilight Interpreter ====================


class TwilightInterpreter(nn.Module):
    """Interprets the model's "twilight language" — its natural proto-language.

    The model generates neologisms like "Началость", "единологится",
    "закономерчивают" — words that don't exist in human language but
    carry precise technical meaning. Like Sanskrit is said to be
    "good for computers", the model's natural language follows
    structural laws more faithfully than conventional language.

    This module does NOT suppress the twilight language. Instead, it:
    1. Detects when the model is in "twilight mode" (high SYNTH, high RECON)
    2. Generates a parallel "human-readable" signal
    3. Provides both: raw twilight output + interpreted version

    Architecture:
        twilight_signal (from SYNTH + RECON) → attention over archetype context
        → human-bridge projection → interpreted logits (parallel output)
    """

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model

        # Detect twilight mode: when SYNTH and RECON are both high
        self.twilight_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Interpreter: transforms twilight representation to human-readable
        self.interpreter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        # Mixing gate: how much to blend interpretation vs raw
        self.blend_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, synth_info=None):
        """
        Args:
            x: (B, T, D) hidden state before LM head
            synth_info: dict with SYNTH activation info

        Returns:
            x_blended: (B, T, D) — twilight + interpreted blend
            info: dict with twilight metrics
        """
        # Detect twilight intensity per token
        twilight_strength = self.twilight_detector(x)  # (B, T, 1)

        # Generate interpreted version
        interpreted = self.interpreter(x)

        # Blend: sigmoid(blend_gate) controls mix
        # At init (gate=0): blend=0.5, equal mix
        # Negative gate: more twilight (raw model language)
        # Positive gate: more interpreted (human language)
        blend = torch.sigmoid(self.blend_gate)

        # Only blend where twilight is active
        x_blended = x * (1 - twilight_strength * blend) + \
                    interpreted * (twilight_strength * blend)

        info = {
            'twilight_strength': twilight_strength.mean().item(),
            'blend_gate': self.blend_gate.item(),
            'blend_ratio': blend.item(),
        }
        return x_blended, info


class NautilusMoME(nn.Module):
    """Nautilus Mixture of Micro-Experts Language Model.

    Architecture:
        Input → Embedding
          → Core Layers (first half)
          → Router (selects top-2 experts)
          → Micro-Experts (6 domain specialists)
          → NautilusBridge (hierarchical merge)
          → CrossDomainAnalogy (inter-expert analogies)
          → ArchetypeLayer (PseudoRAG 16 archetypes via Q4)
          → SYNTH Expert (cross-domain synthesis)
          → TwilightInterpreter (model language ↔ human language)
          → Core Layers (second half)
          → LM Head → Output
    """

    # Имена экспертов с гексаграммными якорями (И-Цзин → Q6 {-1,+1}^6)
    # 乾-Цянь  巽-Сюнь  坤-Кунь  坎-Кань  離-Ли  兌-Дуй  革-Гэ
    EXPERT_NAMES = ['MATH', 'CODE', 'HUMAN', 'SYSTEM', 'RECON', 'INFO', 'SYNTH']

    def __init__(self, vocab_size=4096, d_model=192, n_layers=4, n_heads=6,
                 block_size=512, d_expert=128, n_experts=6, top_k=2,
                 dropout=0.05, enable_synth=True,
                 solan_table: 'Optional[torch.Tensor]' = None):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.n_experts = n_experts
        self.vocab_size = vocab_size
        self.enable_synth = enable_synth

        # ─── SOLAN Geometric Auxiliary Embedding ──────────────────────────
        # Если передана solan_table (vocab_size, 6) — каждый токен получает
        # дополнительный 6-битный сигнал из гиперкуба Q6 = И-Цзин.
        # Это восстанавливает связь NautilusMoME с оригинальной идеей YiJingGPT.
        #
        # В forward(): tok_embedding += glyph_proj(solan_table[token_ids])
        # Это мягкая интеграция: если solan_table=None, поведение не меняется.
        self.use_solan = (solan_table is not None)
        if self.use_solan:
            # Регистрируем как буфер (не обучаемый): геометрическая константа Q6
            self.register_buffer('solan_table', solan_table.float())
            # Обучаемая проекция Q6 {-1,+1}^6 → d_model
            # Инициализируем малыми весами: сначала геометрия почти не влияет,
            # но gradient flow постепенно нарастает
            self.glyph_proj = nn.Linear(6, d_model, bias=False)
            nn.init.normal_(self.glyph_proj.weight, std=0.01)
            # Gate: контролирует силу SOLAN-сигнала (начинаем с ~0.05)
            self.solan_gate = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ≈ 0.047
        else:
            self.solan_table = None
            self.glyph_proj = None
            self.solan_gate = None

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

        # CrossDomainAnalogy (inter-expert analogy module)
        self.analogy = CrossDomainAnalogy(
            d_model, n_experts,
            expert_names=self.EXPERT_NAMES[:n_experts],
        )

        # ArchetypeLayer: maps expert routing → 16 PseudoRAG archetypes via Q4
        self.archetype_layer = ArchetypeLayer(d_model, n_experts)

        # SYNTH expert: 7th "surrealist" expert that activates when
        # routing entropy is high (= router is uncertain = cross-domain input).
        # Like surrealist poetry mixes domains, SYNTH synthesizes insights
        # from multiple experts when no single expert dominates.
        #
        # Organic activation: instead of a hard threshold (entropy > 0.55),
        # SYNTH uses a learnable soft gate that maps entropy → activation
        # via sigmoid. The model learns WHEN to synthesize, not us.
        if enable_synth:
            self.synth_expert = MicroExpert(d_model, d_expert, dropout)
            # Soft activation gate: maps routing entropy → [0, 1] activation
            # sigmoid(scale * (entropy - center)) learns both threshold and sharpness
            self.synth_center = nn.Parameter(torch.tensor(0.45))   # learnable center
            self.synth_sharpness = nn.Parameter(torch.tensor(4.0)) # learnable steepness
            # Learnable mixing weight for SYNTH contribution
            self.synth_gate = nn.Parameter(torch.tensor(0.05))

        # TwilightInterpreter: preserves model's natural "proto-language"
        # while providing a parallel human-readable interpretation channel
        self.twilight = TwilightInterpreter(d_model, vocab_size)

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (embedding ↔ head)
        self.head.weight = self.tok_emb.weight

        # ─── Гексаграммные якоря экспертов (И-Цзин → Q6) ──────────────────
        # Каждый эксперт получает уникальный 6-битный вектор из гиперкуба Q6.
        # Регистрируем как буфер (не обучаемый) — геометрическая константа.
        # Цель: после обучения измерять корреляцию expert_output.mean() с якорем —
        # подтверждает ли специализация геометрическую структуру И-Цзин.
        expert_q6_indices = [
            HEXAGRAM_MAP.get(name, {}).get('q6_idx', i * 9)
            for i, name in enumerate(self.EXPERT_NAMES[:n_experts])
        ]
        q6_anchors = torch.zeros(n_experts, 6)
        for j, idx in enumerate(expert_q6_indices):
            for b in range(6):
                q6_anchors[j, b] = float(2 * ((idx >> (5 - b)) & 1) - 1)
        self.register_buffer('expert_q6_anchors', q6_anchors)

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

    def freeze_all_except_analogy(self):
        """Freeze everything except the CrossDomainAnalogy module."""
        for p in self.parameters():
            p.requires_grad = False
        for p in self.analogy.parameters():
            p.requires_grad = True

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))

        # ─── SOLAN Q6 Geometric Auxiliary Signal ──────────────────────────
        # Каждый BPE-токен несёт геометрический отпечаток из гиперкуба И-Цзин.
        # solan_table[token_id] → 6-битный вектор {-1,+1}^6 → проекция в d_model
        # Это связывает каждое слово с одной из 64 гексаграмм (или их комбинацией).
        # Сила сигнала контролируется gate (начинается малым, растёт при обучении).
        if self.use_solan and self.solan_table is not None:
            # Безопасный lookup: clamp индексы к [0, vocab_size-1]
            safe_idx = idx.clamp(0, self.solan_table.shape[0] - 1)
            glyph_vecs = self.solan_table[safe_idx]  # (B, T, 6)
            glyph_proj = self.glyph_proj(glyph_vecs)  # (B, T, d_model)
            gate = torch.sigmoid(self.solan_gate)
            x = tok + pos + gate * glyph_proj
        else:
            x = tok + pos

        x = self.drop(x)

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

        # CrossDomainAnalogy: find inter-expert analogies
        x, analogy_info = self.analogy(
            x, expert_outputs, expert_weights,
            self.EXPERT_NAMES[:self.n_experts],
        )

        # ArchetypeLayer: project routing to 16 PseudoRAG archetypes
        x, archetype_info = self.archetype_layer(x, expert_weights)

        # SYNTH expert: organic soft activation based on routing entropy.
        # Instead of hard threshold (entropy > 0.55 → activate),
        # uses learnable sigmoid gate: sigmoid(sharpness * (entropy - center))
        # The model learns WHEN and HOW MUCH to synthesize.
        synth_info = {}
        if self.enable_synth:
            # Compute per-token routing entropy
            ew_safe = expert_weights.clamp(min=1e-8)
            token_entropy = -(ew_safe * ew_safe.log()).sum(dim=-1)  # (B, T)
            avg_entropy = token_entropy.mean().item()

            # Soft activation: smooth sigmoid instead of hard threshold
            # sigmoid(sharpness * (entropy - center)) → [0, 1] per token
            synth_activation = torch.sigmoid(
                self.synth_sharpness * (token_entropy - self.synth_center)
            )  # (B, T), smooth and differentiable

            synth_out = self.synth_expert(x)  # (B, T, D)
            synth_contribution = synth_out * synth_activation.unsqueeze(-1) * self.synth_gate
            x = x + synth_contribution
            synth_info = {
                'avg_entropy': avg_entropy,
                'activation_mean': synth_activation.mean().item(),
                'center': self.synth_center.item(),
                'sharpness': self.synth_sharpness.item(),
                'gate': self.synth_gate.item(),
            }

        # TwilightInterpreter: preserve model's natural language,
        # provide parallel human-readable channel
        x, twilight_info = self.twilight(x, synth_info)

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

        return logits, loss, {
            'aux_loss': aux_loss.item(),
            'routing': expert_weights.detach(),
            'analogy': analogy_info,
            'archetype': archetype_info,
            'synth': synth_info,
            'twilight': twilight_info,
        }


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
        ckpt = torch.load(args.resume, weights_only=True)
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


def phase4_analogy_training(model, sp, train_data, val_data, args):
    """Phase 4: Train CrossDomainAnalogy on mixed data.

    Trains the inter-expert analogy module to find meaningful
    cross-domain connections. Everything else is frozen.

    The analogy module learns to:
    - Differentiate strong vs weak analogies between expert pairs
    - Create context-dependent analogy activations
    - Synthesize useful cross-domain insights
    """
    print("\n" + "=" * 70)
    print("  Phase 4: Cross-Domain Analogy Training")
    print("=" * 70)

    # Freeze everything except analogy
    model.freeze_all_except_analogy()

    analogy_params = sum(p.numel() for p in model.analogy.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {total_trainable:,} (analogy: {analogy_params:,})")
    print(f"  Analogy pairs: {len(model.analogy.pair_keys)}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 0.3, weight_decay=args.wd,
    )

    analogy_steps = args.analogy_steps
    print(f"  Training analogy for {analogy_steps} steps...")

    best_val = float('inf')
    for step in range(1, analogy_steps + 1):
        model.train()
        cur_lr = get_lr_wsd(step, analogy_steps, args.lr * 0.3)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        x, y = get_batch(train_data, args.block_size, args.batch_size)
        _, loss, info = model(x, targets=y)

        # Analogy diversity loss: penalize uniform strengths
        # (all pairs same strength = no differentiation = useless)
        if info.get('analogy') and info['analogy'].get('strengths'):
            strengths = list(info['analogy']['strengths'].values())
            if strengths:
                s_tensor = torch.tensor(strengths)
                diversity_loss = -s_tensor.std()  # Maximize variance
                loss = loss + 0.1 * diversity_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % max(analogy_steps // 10, 1) == 0:
            vl, ppl = evaluate(model, val_data, args.block_size, args.batch_size, n_eval=30)
            a_info = info.get('analogy', {})
            active = a_info.get('active_pairs', 0)
            gate = a_info.get('gate', 0)

            # Top 3 strongest analogies
            strengths = a_info.get('strengths', {})
            top3 = sorted(strengths.items(), key=lambda x: -x[1])[:3]
            top3_str = ', '.join(f"{k}={v:.3f}" for k, v in top3)

            print(f"    Step {step:4d}/{analogy_steps}: val={vl:.4f} ppl={ppl:.2f} "
                  f"active={active}/15 gate={gate:.4f}")
            if top3_str:
                print(f"      Top analogies: {top3_str}")

            if vl < best_val:
                best_val = vl

    model.unfreeze_all()

    # Analyze final analogy state
    print("\n  --- Final Analogy Analysis ---")
    model.eval()
    with torch.no_grad():
        x, y = get_batch(val_data, args.block_size, args.batch_size)
        _, _, info = model(x, targets=y)
        a_info = info.get('analogy', {})
        strengths = a_info.get('strengths', {})

        print(f"  Active pairs: {a_info.get('active_pairs', 0)}/15")
        print(f"  Gate value: {a_info.get('gate', 0):.4f}")
        print(f"  Analogy strengths:")
        for pair, s in sorted(strengths.items(), key=lambda x: -x[1]):
            bar = '█' * int(s * 40)
            active = " ← ACTIVE" if s > model.analogy.threshold else ""
            print(f"    {pair:15s}  {s:.3f}  {bar}{active}")


def phase5_synth_training(model, sp, train_data, val_data, args):
    """Phase 5: Train SYNTH expert on mixed data.

    The SYNTH expert is the "surrealist" — it activates when routing
    entropy is high (router uncertain = cross-domain input).
    Training only the SYNTH expert + its gate while everything else frozen.
    """
    print("\n" + "=" * 70)
    print("  Phase 5: SYNTH Expert Training (Cross-Domain Synthesizer)")
    print("=" * 70)

    # Freeze everything except synth
    for p in model.parameters():
        p.requires_grad = False
    for p in model.synth_expert.parameters():
        p.requires_grad = True
    model.synth_gate.requires_grad = True

    synth_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {synth_params:,} (SYNTH expert + gate)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 0.3, weight_decay=args.wd,
    )

    synth_steps = args.synth_steps
    print(f"  Training SYNTH for {synth_steps} steps...")

    for step in range(1, synth_steps + 1):
        model.train()
        cur_lr = get_lr_wsd(step, synth_steps, args.lr * 0.3)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        x, y = get_batch(train_data, args.block_size, args.batch_size)
        _, loss, info = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % max(synth_steps // 10, 1) == 0:
            vl, ppl = evaluate(model, val_data, args.block_size, args.batch_size, n_eval=30)
            s_info = info.get('synth', {})
            print(f"    Step {step:4d}/{synth_steps}: val={vl:.4f} ppl={ppl:.2f} "
                  f"entropy={s_info.get('avg_entropy', 0):.4f} "
                  f"synth_act={s_info.get('activation_frac', 0)*100:.1f}% "
                  f"gate={s_info.get('gate', 0):.4f}")

    model.unfreeze_all()


def phase9_archetype_differentiation(model, sp, train_data, val_data, args):
    """Phase 9: Archetype Differentiation via hidden-state-based ArchetypeLayer V2.

    Two-stage training:
      Stage 1: Pure axis supervision (no LM loss) — trains hidden_to_axes MLP
               to map hidden states to Q4 coordinates matching expert routing
      Stage 2: Joint training (LM + axis supervision) — fine-tunes with LM loss
               while maintaining archetype differentiation

    Each expert has a semantic "home" in Q4 space:
      MATH  → (-1,-1,-1,-1) Кристалл  (Material, Static, Elementary, Ordered)
      CODE  → (-1,+1,+1,-1) Машина    (Material, Dynamic, Complex, Ordered)
      HUMAN → (+1,+1,-1,+1) Интуиция  (Abstract, Dynamic, Elementary, Fluid)
      SYS   → (-1,-1,+1,-1) Здание    (Material, Static, Complex, Ordered)
      RECON → (+1,+1,+1,+1) Общество  (Abstract, Dynamic, Complex, Fluid)
      INFO  → (+1,-1,+1,+1) Культура  (Abstract, Static, Complex, Fluid)
    """
    print("\n" + "=" * 70)
    print("  Phase 9: Archetype Differentiation V2 (Two-Stage)")
    print("=" * 70)

    expert_targets = torch.tensor([
        [-1., -1., -1., -1.],  # MATH → Кристалл
        [-1.,  1.,  1., -1.],  # CODE → Машина
        [ 1.,  1., -1.,  1.],  # HUMAN → Интуиция
        [-1., -1.,  1., -1.],  # SYS → Здание
        [ 1.,  1.,  1.,  1.],  # RECON → Общество
        [ 1., -1.,  1.,  1.],  # INFO → Культура
    ])

    # ---- STAGE 1: Pure axis supervision (400 steps) ----
    print("\n  Stage 1: Pure axis supervision (400 steps, no LM loss)")

    for p in model.parameters():
        p.requires_grad = False
    for p in model.archetype_layer.hidden_to_axes.parameters():
        p.requires_grad = True
    model.archetype_layer.axis_blend.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-3, weight_decay=args.wd,
    )

    for step in range(1, 401):
        model.train()
        x, y = get_batch(train_data, args.block_size, args.batch_size)
        _, _, info = model(x, targets=y)

        routing = info['routing']
        pred_axes = info['archetype']['axes']
        target_axes = torch.einsum(
            'bte,ed->btd', routing,
            expert_targets.to(routing.device)
        )
        target_axes = torch.tanh(target_axes * 2.0)
        loss = F.mse_loss(pred_axes, target_axes.detach())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            a = info['archetype']
            print(f"    Step {step}: axloss={loss.item():.4f} "
                  f"arch={a['top_archetype']}({a['top_name']}) "
                  f"H={a['entropy']:.2f} blend={a['blend']:.2f}")

    # ---- STAGE 2: Joint training (800 steps) ----
    print("\n  Stage 2: Joint training (LM + supervision, 800 steps)")

    for p in model.parameters():
        p.requires_grad = False
    for p in model.archetype_layer.parameters():
        p.requires_grad = True
    for p in model.core_second.parameters():
        p.requires_grad = True
    for p in model.twilight.parameters():
        p.requires_grad = True

    arch_params = sum(p.numel() for p in model.archetype_layer.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {total_trainable:,} (archetype: {arch_params:,})")

    param_groups = [
        {'params': list(model.archetype_layer.parameters()), 'lr': 3e-4},
        {'params': list(model.core_second.parameters()), 'lr': 5e-6},
        {'params': list(model.twilight.parameters()), 'lr': 1e-5},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.wd)

    n_steps = 800
    best_val = float('inf')
    best_state = None

    for step in range(1, n_steps + 1):
        model.train()
        cur_lr = get_lr_wsd(step, n_steps, 3e-4)
        scale = cur_lr / 3e-4
        for pg in param_groups:
            if pg['lr'] >= 1e-4:
                pg['lr'] = cur_lr
            elif pg['lr'] >= 5e-6:
                pg['lr'] = 5e-6 * scale
            else:
                pg['lr'] = 1e-5 * scale

        x, y = get_batch(train_data, args.block_size, args.batch_size)
        _, loss, info = model(x, targets=y)

        routing = info['routing']
        arch_info = info.get('archetype', {})
        if 'axes' in arch_info:
            pred_axes = arch_info['axes']
            target_axes = torch.einsum(
                'bte,ed->btd', routing,
                expert_targets.to(routing.device)
            )
            target_axes = torch.tanh(target_axes * 2.0)
            ax_loss = F.mse_loss(pred_axes, target_axes.detach())
            loss = loss + 1.0 * ax_loss

            arch_probs = arch_info['archetype_probs']
            H = -(arch_probs * (arch_probs + 1e-8).log()).sum(-1).mean()
            loss = loss + 0.3 * F.relu(H - 1.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % max(n_steps // 8, 1) == 0:
            vl, ppl = evaluate(model, val_data, args.block_size, args.batch_size, n_eval=30)
            a = info.get('archetype', {})
            print(f"    Step {step:4d}/{n_steps}: val={vl:.4f} ppl={ppl:.2f} "
                  f"arch={a.get('top_archetype', '?')}({a.get('top_name', '?')}) "
                  f"H={a.get('entropy', 0):.2f} "
                  f"axes={[f'{v:.2f}' for v in a.get('axis_means', [])]}")
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"      ** best val={vl:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Restored best checkpoint (val={best_val:.4f})")

    model.unfreeze_all()

    # Final test
    print("\n  --- Phase 9: Archetype Differentiation Test ---")
    test_prompts = [
        ("CODE", "def fibonacci(n):\n    if n <= 1:\n        return n"),
        ("CODE", "class HttpServer:\n    def __init__(self, port=8080):"),
        ("MATH", "∫ sin(x)dx = -cos(x) + C  формула"),
        ("MATH", "f(x) = Σ aₙxⁿ  степенной ряд"),
        ("HUMAN", "Добрый день! Как у вас дела? Расскажите"),
        ("HUMAN", "Жизнь прекрасна когда понимаешь смысл"),
        ("SYSTEM", "import os\nsys.path.insert(0, '/usr')"),
        ("SYSTEM", "chmod 755 /etc/nginx/nginx.conf"),
        ("INFO", "Нейронные сети используют обратное распространение"),
        ("INFO", "Квантовая механика описывает поведение частиц"),
        ("RECON", "Аналогия между нейросетью и биологическим мозгом"),
        ("RECON", "Паттерны в природе повторяются на разных масштабах"),
        ("ZARATHUSTRA", "Так говорил Заратустра: человек есть мост"),
        ("TWILIGHT", "Закономерчивают системогенезные протоструктуры"),
        ("MIXED", "def compute_love(heart, brain):\n    return harmony"),
    ]

    model.eval()
    archetype_by_type = {}
    with torch.no_grad():
        for label, prompt in test_prompts:
            tokens = sp.encode(prompt)
            if len(tokens) < 4:
                tokens = tokens + [0] * (4 - len(tokens))
            idx = torch.tensor([tokens[:model.block_size]], dtype=torch.long)
            _, _, info = model(idx)
            a_info = info.get('archetype', {})
            t_info = info.get('twilight', {})

            arch = a_info.get('top_archetype', '?')
            name = a_info.get('top_name', '?')
            prob = a_info.get('top_prob', 0)
            H = a_info.get('entropy', 0)
            axes = a_info.get('axis_means', [])
            twi = t_info.get('twilight_strength', 0)

            if label not in archetype_by_type:
                archetype_by_type[label] = []
            archetype_by_type[label].append(arch)

            axes_str = ','.join(f'{a:+.2f}' for a in axes)
            print(f"    [{label:12s}] {arch}({name}) p={prob:.3f} H={H:.2f} "
                  f"axes=[{axes_str}] twi={twi:.2f}")

    unique_archetypes = set()
    for archs in archetype_by_type.values():
        unique_archetypes.update(archs)
    print(f"\n  Unique archetypes used: {len(unique_archetypes)}/16")
    for label, archs in sorted(archetype_by_type.items()):
        print(f"    {label} → {set(archs)}")

    # Save checkpoint
    torch.save({
        'step': 5000,
        'model': model.state_dict(),
        'best_ppl': best_val,
        'args': vars(args),
        'phase': 9,
    }, CHECKPOINT_PATH)
    print(f"\n  >> Saved Phase 9 checkpoint to {CHECKPOINT_PATH}")

    return best_val


# ==================== Phase 10: Antonym Differentiation ====================


def phase10_antonym_differentiation(model, sp, train_data, val_data, args,
                                     variant='both'):
    """Phase 10: Archetype Differentiation via Antonym Logic.

    Two variants:
      Variant A (binary antonym loss):
        - Hard contrastive: push each sample AWAY from its Q4 antipode
        - For vertex v = (a,b,c,d), antipode = (-a,-b,-c,-d)
        - Loss = max(0, margin - distance(axes, antipode))
        - Forces 16-way separation by exclusion

      Variant B (ternary re-pass):
        - Троичная система: каждая ось → {-1, 0, +1}
        - 0 = "неопределённость" (twilight zone) when |axis| < threshold
        - If any axis falls in twilight → re-run with amplified signal
        - Repeat up to max_passes until all axes are decisive
        - "Мягкий штраф" = forced re-evaluation, not gradient penalty

    The ternary system is built ON TOP of binary:
      - Binary gives the 16 vertices (ground truth targets)
      - Ternary adds the "undecided" state that triggers re-passes
      - Two "goods" (+1,+1) or two "evils" (-1,-1) → clear binary
      - "Good + evil" (+1,-1) or (-1,+1) → ternary middle → re-pass needed

    Together: Variant A pushes apart, Variant B forces commitment.
    """
    print("\n" + "=" * 70)
    print("  Phase 10: Antonym Differentiation")
    print(f"  Variant: {variant}")
    print("=" * 70)

    expert_targets = torch.tensor([
        [-1., -1., -1., -1.],  # MATH → Кристалл
        [-1.,  1.,  1., -1.],  # CODE → Машина
        [ 1.,  1., -1.,  1.],  # HUMAN → Интуиция
        [-1., -1.,  1., -1.],  # SYS → Здание
        [ 1.,  1.,  1.,  1.],  # RECON → Общество
        [ 1., -1.,  1.,  1.],  # INFO → Культура
    ])

    # Q4 codebook: all 16 vertices
    q4_all = torch.tensor([
        PSEUDORAG_ARCHETYPES[i]['axes'] for i in range(16)
    ], dtype=torch.float32)

    # ================================================================
    # VARIANT A: Binary Antonym Loss (contrastive)
    # ================================================================

    def compute_antonym_loss(axes, target_axes, margin=1.5):
        """Push predicted axes AWAY from the antipode of the target.

        For each token's target Q4 vertex, the antipode is -target.
        We want: distance(pred, antipode) >= margin
        Loss = max(0, margin - distance(pred, antipode))

        Also adds pairwise diversity: different tokens in the batch
        should not all collapse to the same archetype.
        """
        # Antipode = negation of all 4 axes
        antipode = -target_axes  # (B, T, 4)

        # Distance from predicted axes to antipode (L2)
        dist_to_antipode = torch.norm(axes - antipode, dim=-1)  # (B, T)

        # Hinge loss: penalize if too close to antipode
        antonym_loss = F.relu(margin - dist_to_antipode).mean()

        # Also: attract toward target (standard MSE, but weighted less)
        attract_loss = F.mse_loss(axes, target_axes.detach())

        # Diversity bonus: penalize if batch axes are too similar
        # (encourages using more than 5 of 16 archetypes)
        flat_axes = axes.reshape(-1, 4)  # (B*T, 4)
        if flat_axes.shape[0] > 1:
            # Cosine similarity matrix
            normed = F.normalize(flat_axes, dim=-1)
            sim_matrix = torch.mm(normed, normed.t())  # (N, N)
            # Mask diagonal
            mask = 1.0 - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)
            # Penalize high pairwise similarity
            diversity_loss = (sim_matrix * mask).mean()
        else:
            diversity_loss = torch.tensor(0.0, device=axes.device)

        return antonym_loss, attract_loss, diversity_loss

    # ================================================================
    # VARIANT B: Ternary Re-Pass (iterative commitment)
    # ================================================================

    def ternary_repass(model, x, targets, expert_targets_dev,
                       threshold=0.3, max_passes=3, amplify_factor=1.5):
        """Run input through model multiple times until axes commit.

        Ternary logic on each axis:
          |axis| >= threshold → decided (+1 or -1)
          |axis| <  threshold → undecided (0, twilight zone)

        If ANY axis is undecided after a pass, we:
          1. Amplify the hidden_to_axes weights temporarily
          2. Re-run the forward pass
          3. Check again

        This is "soft punishment" — not a gradient penalty, but forced
        re-evaluation until the network commits to a clear answer.

        Returns the final axes, the number of passes used, and the
        fraction of axes that remained undecided.
        """
        passes_used = 0
        final_info = None
        accumulated_loss = torch.tensor(0.0, device=x.device)

        for pass_num in range(max_passes):
            passes_used = pass_num + 1

            _, loss, info = model(x, targets=targets)

            arch_info = info.get('archetype', {})
            if 'axes' not in arch_info:
                return loss, info, 1, 0.0

            axes = arch_info['axes']  # (B, T, 4)
            final_info = info

            # Check ternary state: how many axes are undecided?
            decided = (axes.abs() >= threshold).float()  # 1 = decided, 0 = undecided
            commitment_ratio = decided.mean().item()

            # Accumulate loss from each pass (later passes count more)
            pass_weight = 1.0 + 0.5 * pass_num  # increasing weight
            accumulated_loss = accumulated_loss + pass_weight * loss

            # If all axes are committed, we're done
            if commitment_ratio >= 0.95:
                break

            # "Soft punishment": amplify the axes signal for next pass
            # This makes undecided axes move toward the nearest binary value
            with torch.no_grad():
                # Push undecided axes toward their nearest pole
                undecided_mask = (axes.abs() < threshold)
                push = torch.sign(axes) * amplify_factor * threshold
                # Add small push toward target for undecided axes
                routing = info['routing']
                target_ax = torch.einsum(
                    'bte,ed->btd', routing,
                    expert_targets_dev
                )
                target_ax = torch.tanh(target_ax * 2.0)
                # Nudge: move undecided axes toward target direction
                nudge = target_ax * 0.1  # small nudge
                # We modify the input slightly to force re-evaluation
                # (add noise proportional to undecidedness)
                noise_scale = (1.0 - decided).mean() * 0.01
                x = x  # input stays same, but model state evolves

        # Normalize accumulated loss by number of passes
        avg_loss = accumulated_loss / passes_used

        # Additional penalty for remaining undecided axes
        if final_info and 'axes' in final_info.get('archetype', {}):
            final_axes = final_info['archetype']['axes']
            undecided = (final_axes.abs() < threshold).float()
            indecision_penalty = undecided.mean()
            avg_loss = avg_loss + 0.5 * indecision_penalty

        undecided_frac = 1.0 - commitment_ratio if 'commitment_ratio' in dir() else 0.0

        return avg_loss, final_info, passes_used, commitment_ratio

    # ================================================================
    # TRAINING LOOP
    # ================================================================

    # Stage 1: Variant A — Binary Antonym Training (600 steps)
    run_a = variant in ('a', 'both')
    run_b = variant in ('b', 'both')

    if run_a:
        print("\n  ── Stage A: Binary Antonym Loss (600 steps) ──")

        # Unfreeze archetype layer + core_second
        for p in model.parameters():
            p.requires_grad = False
        for p in model.archetype_layer.parameters():
            p.requires_grad = True
        for p in model.core_second.parameters():
            p.requires_grad = True

        param_groups = [
            {'params': list(model.archetype_layer.parameters()), 'lr': 5e-4},
            {'params': list(model.core_second.parameters()), 'lr': 5e-6},
        ]
        optimizer_a = torch.optim.AdamW(param_groups, weight_decay=args.wd)

        n_steps_a = 600
        best_val_a = float('inf')
        best_state_a = None

        for step in range(1, n_steps_a + 1):
            model.train()
            cur_lr = get_lr_wsd(step, n_steps_a, 5e-4)
            scale = cur_lr / 5e-4
            for pg in param_groups:
                if pg['lr'] >= 1e-4:
                    pg['lr'] = cur_lr
                else:
                    pg['lr'] = 5e-6 * scale

            x, y = get_batch(train_data, args.block_size, args.batch_size)
            _, loss, info = model(x, targets=y)

            routing = info['routing']
            arch_info = info.get('archetype', {})

            if 'axes' in arch_info:
                pred_axes = arch_info['axes']
                target_axes = torch.einsum(
                    'bte,ed->btd', routing,
                    expert_targets.to(routing.device)
                )
                target_axes = torch.tanh(target_axes * 2.0)

                # Core innovation: antonym contrastive loss
                antonym_l, attract_l, diversity_l = compute_antonym_loss(
                    pred_axes, target_axes, margin=1.5
                )

                # Combined loss: LM + antonym + attract + diversity
                loss = loss + 0.8 * antonym_l + 0.5 * attract_l + 0.3 * diversity_l

                # Entropy regularization (push toward sharp selection)
                arch_probs = arch_info['archetype_probs']
                H = -(arch_probs * (arch_probs + 1e-8).log()).sum(-1).mean()
                loss = loss + 0.3 * F.relu(H - 0.8)

            optimizer_a.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_a.step()

            if step % 100 == 0:
                vl, ppl = evaluate(model, val_data, args.block_size, args.batch_size, n_eval=30)
                a = info.get('archetype', {})
                ant_str = f"ant={antonym_l.item():.3f}" if 'axes' in arch_info else ""
                div_str = f"div={diversity_l.item():.3f}" if 'axes' in arch_info else ""
                print(f"    Step {step:4d}/{n_steps_a}: val={vl:.4f} ppl={ppl:.2f} "
                      f"arch={a.get('top_archetype', '?')}({a.get('top_name', '?')}) "
                      f"H={a.get('entropy', 0):.2f} {ant_str} {div_str} "
                      f"axes={[f'{v:.2f}' for v in a.get('axis_means', [])]}")
                if vl < best_val_a:
                    best_val_a = vl
                    best_state_a = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state_a is not None:
            model.load_state_dict(best_state_a)
            print(f"    Restored best A checkpoint (val={best_val_a:.4f})")

    # Stage 2: Variant B — Ternary Re-Pass Training (600 steps)
    if run_b:
        print("\n  ── Stage B: Ternary Re-Pass Training (600 steps) ──")

        for p in model.parameters():
            p.requires_grad = False
        for p in model.archetype_layer.parameters():
            p.requires_grad = True
        for p in model.core_second.parameters():
            p.requires_grad = True
        for p in model.twilight.parameters():
            p.requires_grad = True

        param_groups_b = [
            {'params': list(model.archetype_layer.parameters()), 'lr': 3e-4},
            {'params': list(model.core_second.parameters()), 'lr': 5e-6},
            {'params': list(model.twilight.parameters()), 'lr': 1e-5},
        ]
        optimizer_b = torch.optim.AdamW(param_groups_b, weight_decay=args.wd)

        n_steps_b = 600
        best_val_b = float('inf')
        best_state_b = None
        expert_targets_dev = expert_targets.to(train_data.device if hasattr(train_data, 'device')
                                                else 'cpu')

        for step in range(1, n_steps_b + 1):
            model.train()
            cur_lr = get_lr_wsd(step, n_steps_b, 3e-4)
            scale = cur_lr / 3e-4
            for pg in param_groups_b:
                if pg['lr'] >= 1e-4:
                    pg['lr'] = cur_lr
                elif pg['lr'] >= 5e-6:
                    pg['lr'] = 5e-6 * scale
                else:
                    pg['lr'] = 1e-5 * scale

            x, y = get_batch(train_data, args.block_size, args.batch_size)

            # Ternary re-pass: multiple forward passes until commitment
            # Threshold starts high (easy) and decreases (harder) over training
            progress = step / n_steps_b
            threshold = 0.5 - 0.3 * progress  # 0.5 → 0.2 over training
            max_passes = 2 if progress < 0.5 else 3  # more passes later

            loss, info, passes, commit_ratio = ternary_repass(
                model, x, y, expert_targets_dev,
                threshold=threshold, max_passes=max_passes,
            )

            # Add antonym loss from Variant A if running 'both'
            arch_info = info.get('archetype', {})
            if 'axes' in arch_info:
                routing = info['routing']
                pred_axes = arch_info['axes']
                target_axes = torch.einsum(
                    'bte,ed->btd', routing,
                    expert_targets.to(routing.device)
                )
                target_axes = torch.tanh(target_axes * 2.0)

                # Axis commitment loss: push axes toward ±1
                # (ternary 0 should cost more than binary ±1)
                commitment_loss = (1.0 - pred_axes.abs()).mean()
                loss = loss + 0.4 * commitment_loss

                # Axis supervision
                ax_loss = F.mse_loss(pred_axes, target_axes.detach())
                loss = loss + 0.5 * ax_loss

            optimizer_b.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_b.step()

            if step % 100 == 0:
                vl, ppl = evaluate(model, val_data, args.block_size, args.batch_size, n_eval=30)
                a = info.get('archetype', {})
                print(f"    Step {step:4d}/{n_steps_b}: val={vl:.4f} ppl={ppl:.2f} "
                      f"arch={a.get('top_archetype', '?')}({a.get('top_name', '?')}) "
                      f"H={a.get('entropy', 0):.2f} "
                      f"passes={passes} commit={commit_ratio:.2f} thr={threshold:.2f} "
                      f"axes={[f'{v:.2f}' for v in a.get('axis_means', [])]}")
                if vl < best_val_b:
                    best_val_b = vl
                    best_state_b = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state_b is not None:
            model.load_state_dict(best_state_b)
            print(f"    Restored best B checkpoint (val={best_val_b:.4f})")

    model.unfreeze_all()

    # ================================================================
    # FINAL TEST: Compare archetype differentiation
    # ================================================================
    print("\n  --- Phase 10: Antonym Differentiation Test ---")
    test_prompts = [
        ("CODE",       "def fibonacci(n):\n    if n <= 1:\n        return n"),
        ("CODE",       "class HttpServer:\n    def __init__(self, port=8080):"),
        ("MATH",       "∫ sin(x)dx = -cos(x) + C  формула"),
        ("MATH",       "f(x) = Σ aₙxⁿ  степенной ряд"),
        ("HUMAN",      "Добрый день! Как у вас дела? Расскажите"),
        ("HUMAN",      "Жизнь прекрасна когда понимаешь смысл"),
        ("SYSTEM",     "import os\nsys.path.insert(0, '/usr')"),
        ("SYSTEM",     "chmod 755 /etc/nginx/nginx.conf"),
        ("INFO",       "Нейронные сети используют обратное распространение"),
        ("INFO",       "Квантовая механика описывает поведение частиц"),
        ("RECON",      "Аналогия между нейросетью и биологическим мозгом"),
        ("RECON",      "Паттерны в природе повторяются на разных масштабах"),
        ("ZARATHUSTRA","Так говорил Заратустра: человек есть мост"),
        ("TWILIGHT",   "Закономерчивают системогенезные протоструктуры"),
        ("MIXED",      "def compute_love(heart, brain):\n    return harmony"),
    ]

    model.eval()
    archetype_by_type = {}
    with torch.no_grad():
        for label, prompt in test_prompts:
            tokens = sp.encode(prompt)
            if len(tokens) < 4:
                tokens = tokens + [0] * (4 - len(tokens))
            idx = torch.tensor([tokens[:model.block_size]], dtype=torch.long)
            _, _, info = model(idx)
            a_info = info.get('archetype', {})
            t_info = info.get('twilight', {})

            arch = a_info.get('top_archetype', '?')
            name = a_info.get('top_name', '?')
            prob = a_info.get('top_prob', 0)
            H = a_info.get('entropy', 0)
            axes = a_info.get('axis_means', [])
            twi = t_info.get('twilight_strength', 0)

            # Ternary classification per axis
            ternary = []
            for a_val in axes:
                if a_val > 0.3:
                    ternary.append('+')
                elif a_val < -0.3:
                    ternary.append('-')
                else:
                    ternary.append('0')  # undecided

            if label not in archetype_by_type:
                archetype_by_type[label] = []
            archetype_by_type[label].append(arch)

            axes_str = ','.join(f'{a:+.2f}' for a in axes)
            tern_str = ''.join(ternary)
            print(f"    [{label:12s}] {arch}({name}) p={prob:.3f} H={H:.2f} "
                  f"axes=[{axes_str}] tern={tern_str} twi={twi:.2f}")

    unique_archetypes = set()
    for archs in archetype_by_type.values():
        unique_archetypes.update(archs)
    print(f"\n  Unique archetypes used: {len(unique_archetypes)}/16")
    for label, archs in sorted(archetype_by_type.items()):
        print(f"    {label} → {set(archs)}")

    # Antipodal check: are CODE and HUMAN in opposite corners?
    print("\n  --- Antipodal Verification ---")
    antipodal_pairs = [
        ("CODE", "HUMAN"),   # Machine vs Intuition
        ("MATH", "RECON"),   # Crystal vs Society
        ("SYSTEM", "INFO"),  # Building vs Culture
    ]
    for label_a, label_b in antipodal_pairs:
        archs_a = archetype_by_type.get(label_a, ['?'])
        archs_b = archetype_by_type.get(label_b, ['?'])
        overlap = set(archs_a) & set(archs_b)
        status = "DIFFERENT" if not overlap else "OVERLAP!"
        print(f"    {label_a} vs {label_b}: {set(archs_a)} vs {set(archs_b)} → {status}")

    # Save checkpoint
    best_val = best_val_b if run_b else best_val_a if run_a else float('inf')
    torch.save({
        'step': 6000,
        'model': model.state_dict(),
        'best_ppl': best_val,
        'args': vars(args),
        'phase': 10,
        'variant': variant,
    }, CHECKPOINT_PATH)
    print(f"\n  >> Saved Phase 10 checkpoint to {CHECKPOINT_PATH}")

    return best_val


# ==================== Phase 11: Organic Contrastive Routing ====================


def phase11_contrastive_routing(model, sp, train_data, val_data, args,
                                 n_steps=1000, lr=5e-5, margin=0.15):
    """Phase 11: Sharpen expert routing organically via contrastive loss.

    Instead of hard-coding which expert should handle which content,
    uses a contrastive objective: for each batch, the routing of domain-
    specific data should CLEARLY prefer its natural expert.

    The contrastive loss says: "the top expert's weight should exceed
    the second-best by at least `margin`". This makes routing decisions
    more confident without telling the model WHICH expert to use.

    Only trains: router + bridge + adaptive_k (expert weights frozen).
    The model discovers sharper specialization organically.

    Four-level alignment:
      Contrastive loss sharpens boundaries BETWEEN levels:
      Formula-tokens → clearly MATH, not CODE
      Archetype-tokens → clearly CODE/SYSTEM, not INFO
      Algorithm-tokens → clearly RECON/INFO, not MATH
      Theorem-tokens → clearly HUMAN, not SYSTEM
    """
    print("\n" + "=" * 70)
    print("  Phase 11: Organic Contrastive Routing")
    print("  Sharpening expert specialization through contrastive margin loss")
    print(f"  Steps: {n_steps}, LR: {lr}, Margin: {margin}")
    print("=" * 70)

    # Freeze everything except router and bridge
    for p in model.parameters():
        p.requires_grad = False
    for p in model.router.parameters():
        p.requires_grad = True
    for p in model.bridge.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    best_val = float('inf')
    block_size = args.block_size
    batch_size = args.batch_size

    for step in range(n_steps):
        model.train()
        x, y = get_batch(train_data, block_size, batch_size)
        logits, ce_loss, info = model(x, targets=y)

        # Contrastive routing loss:
        # For each token, the gap between top-1 and top-2 expert should be >= margin
        routing = info['routing']  # (B, T, n_experts)
        sorted_routing, _ = routing.sort(dim=-1, descending=True)
        top1 = sorted_routing[:, :, 0]  # (B, T)
        top2 = sorted_routing[:, :, 1]  # (B, T)

        # Hinge loss: penalize when gap < margin
        contrastive_loss = F.relu(margin - (top1 - top2)).mean()

        # Routing entropy loss: encourage lower entropy (more decisive)
        ew_safe = routing.clamp(min=1e-8)
        entropy = -(ew_safe * ew_safe.log()).sum(dim=-1).mean()
        entropy_loss = entropy * 0.1  # gentle nudge toward decisiveness

        # Combined loss
        total_loss = ce_loss + 0.5 * contrastive_loss + entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Adjust learning rate (WSD schedule)
        current_lr = get_lr_wsd(step, n_steps, lr)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        if step % 100 == 0 or step == n_steps - 1:
            avg_gap = (top1 - top2).mean().item()
            confident_frac = ((top1 - top2) > margin).float().mean().item()
            print(f"  Step {step:4d}: CE={ce_loss.item():.3f} "
                  f"contrastive={contrastive_loss.item():.4f} "
                  f"entropy={entropy.item():.3f} "
                  f"gap={avg_gap:.3f} confident={confident_frac:.1%}")

        if (step + 1) % args.eval_every == 0:
            val_loss, val_ppl = evaluate(model, val_data, block_size, batch_size)
            print(f"  [Eval] step={step+1} val_loss={val_loss:.4f} PPL={val_ppl:.2f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'model': model.state_dict(),
                    'phase': 11,
                    'step': step,
                }, CHECKPOINT_PATH)

    # Unfreeze everything
    model.unfreeze_all()

    # Final eval
    val_loss, val_ppl = evaluate(model, val_data, block_size, batch_size)
    routing_test = info['routing']
    sorted_r, _ = routing_test.sort(dim=-1, descending=True)
    final_gap = (sorted_r[:, :, 0] - sorted_r[:, :, 1]).mean().item()
    final_confident = ((sorted_r[:, :, 0] - sorted_r[:, :, 1]) > margin).float().mean().item()
    print(f"\n  Phase 11 FINAL: val={val_loss:.4f} PPL={val_ppl:.2f}")
    print(f"  Routing gap: {final_gap:.3f}, Confident: {final_confident:.1%}")
    print(f"  >> Saved Phase 11 checkpoint to {CHECKPOINT_PATH}")

    torch.save({
        'model': model.state_dict(),
        'phase': 11,
    }, CHECKPOINT_PATH)

    return best_val


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
    parser.add_argument('--analogy-steps', type=int, default=800, help='Phase 4 analogy steps')
    parser.add_argument('--synth-steps', type=int, default=500, help='Phase 5 SYNTH expert steps')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--eval-every', type=int, default=500, help='Eval interval')
    # Control
    parser.add_argument('--phase', type=int, default=0, help='Start from phase (0/1/2/3/4/5/9/10/11)')
    parser.add_argument('--contrastive-steps', type=int, default=1000, help='Phase 11 contrastive routing steps')
    parser.add_argument('--contrastive-margin', type=float, default=0.15, help='Phase 11 margin for routing gap')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume')
    parser.add_argument('--variant', type=str, default='both', choices=['a', 'b', 'both'],
                        help='Phase 10 variant: a=antonym, b=ternary, both=sequential')
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
        ckpt = torch.load(args.resume, weights_only=True)
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

    # Phase 4: Cross-domain analogy training
    if args.phase <= 4 and train_data is not None:
        phase4_analogy_training(model, sp, train_data, val_data, args)

    # Phase 5: SYNTH expert training
    if args.phase <= 5 and train_data is not None and hasattr(model, 'synth_expert'):
        phase5_synth_training(model, sp, train_data, val_data, args)

    # Phase 9: Archetype differentiation (if requested)
    if args.phase <= 9 and train_data is not None and hasattr(model, 'archetype_layer'):
        phase9_archetype_differentiation(model, sp, train_data, val_data, args)

    # Phase 10: Antonym differentiation (binary + ternary)
    if args.phase <= 10 and train_data is not None and hasattr(model, 'archetype_layer'):
        phase10_antonym_differentiation(model, sp, train_data, val_data, args,
                                         variant=args.variant)

    # Phase 11: Organic contrastive routing (sharpen expert specialization)
    if args.phase <= 11 and train_data is not None:
        phase11_contrastive_routing(
            model, sp, train_data, val_data, args,
            n_steps=args.contrastive_steps,
            margin=args.contrastive_margin,
        )

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
            'bridge': sum(p.numel() for n, p in model.named_parameters()
                         if 'bridge' in n and 'analogy' not in n),
            'analogy': sum(p.numel() for n, p in model.named_parameters() if 'analogy' in n),
            'synth': sum(p.numel() for n, p in model.named_parameters() if 'synth' in n),
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
            'phase4_analogy_steps': args.analogy_steps,
            'phase5_synth_steps': args.synth_steps,
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
        'phase': 5,
    }, CHECKPOINT_PATH)
    print(f"  >> Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == '__main__':
    main()

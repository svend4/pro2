"""
variant3_extensions.py — Extensions to the Variant3 Q6-Hypercube Architecture

Implements all 10 research ideas plus modules derived from the antonym/conveyor/
Forth-stack/own-alien text analysis:

IDEAS:
  1. HexagramPositionalEncoding  — BFS-distance positions on Q6 graph
  2. SixLineAttention            — 6 heads = 6 lines = 6 domains
  3. BianGuaOptimizer            — optimizer using Q6-step curriculum
  4. TernaryKVCache              — {-1,0,+1} key compression (2-bit)
  5. HexagramTokenizer           — 64-token vocab with biangua merge rules
  6. CrossDomainRAG              — Q6-signature retrieval
  7. HexagramEvaluator           — hex_entropy / domain_coherence metrics
  8. MultiScaleQ6                — Matryoshka Q2→Q3→Q6 hierarchy
  9. AdaptiveHammingScheduler    — λ curriculum warmup/anneal/steady
 10. HexagramMoE                 — 64 experts routed via hex_weights

TEXT-ANALYSIS MODULES:
  BinaryOppositionTable          — antonym axes mapped to Q6 bits
  SvoyChuzhoiGate                — own/alien gate: Q6-distance → +1/0/-1
  BinaryExclusionClassifier      — method-of-exclusion per-axis AND
  TextQualityFilter              — 6-axis quality hexagram
  ConveyorVariant3Block          — named 6-stage pipeline with inspection
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_hexagrams() -> Tensor:
    """All 64 vertices of {-1,+1}^6."""
    verts = list(itertools.product([-1.0, 1.0], repeat=6))
    return torch.tensor(verts, dtype=torch.float32)  # (64, 6)


def _make_biangua_matrix(hexagrams: Tensor) -> Tensor:
    """M[i,j]=1 iff hamming(i,j)==1  ↔  dot==4."""
    dot = hexagrams @ hexagrams.T
    return (dot == 4.0).float()  # (64, 64)


_HEX_CACHE: Optional[Tensor] = None
_BIAN_CACHE: Optional[Tensor] = None


def get_hexagrams() -> Tensor:
    global _HEX_CACHE
    if _HEX_CACHE is None:
        _HEX_CACHE = _make_hexagrams()
    return _HEX_CACHE


def get_biangua() -> Tensor:
    global _BIAN_CACHE
    if _BIAN_CACHE is None:
        _BIAN_CACHE = _make_biangua_matrix(get_hexagrams())
    return _BIAN_CACHE


def bfs_distances(biangua: Tensor) -> Tensor:
    """Compute all-pairs BFS distances on the biangua graph. O(64²)."""
    n = biangua.shape[0]
    adj = (biangua > 0.5).cpu().numpy()
    dist = torch.full((n, n), fill_value=999, dtype=torch.long)
    for src in range(n):
        dist[src, src] = 0
        q: deque = deque([src])
        while q:
            u = q.popleft()
            for v in range(n):
                if adj[u, v] and dist[src, v] == 999:
                    dist[src, v] = dist[src, u] + 1
                    q.append(v)
    return dist  # (64, 64) max value = 6 (diameter of Q6)


# ---------------------------------------------------------------------------
# Idea 1 — HexagramPositionalEncoding
# ---------------------------------------------------------------------------

class HexagramPositionalEncoding(nn.Module):
    """
    Positional encoding derived from BFS distances on the Q6 biangua graph.

    Each sequence position t maps to hexagram index (t % 64).
    The embedding is the distance profile from that hexagram to all 64 others,
    projected into d_model space.

    This gives positions a geometric structure: nearby positions (in sequence)
    that happen to map to biangua-adjacent hexagrams share similar encodings.
    """

    def __init__(self, d_model: int, block_size: int = 512):
        super().__init__()
        hexagrams = get_hexagrams()
        biangua = get_biangua()
        # (64, 64) BFS distances — used as positional features
        dist = bfs_distances(biangua).float()  # values in [0, 6]
        dist_norm = dist / 6.0  # normalise to [0, 1]

        self.register_buffer("dist_norm", dist_norm)  # (64, 64)
        self.proj = nn.Linear(64, d_model, bias=False)
        self.block_size = block_size

    def forward(self, seq_len: int) -> Tensor:
        """Return (seq_len, d_model) positional embeddings."""
        indices = torch.arange(seq_len, device=self.proj.weight.device) % 64
        features = self.dist_norm[indices]  # (T, 64)
        return self.proj(features)  # (T, d_model)


# ---------------------------------------------------------------------------
# Idea 2 — SixLineAttention
# ---------------------------------------------------------------------------

_DOMAIN_ANCHORS = {"GEO": 0, "HYDRO": 18, "PYRO": 45, "AERO": 6, "COSMO": 27, "NOOS": 63}
DOMAIN_NAMES = ["GEO", "HYDRO", "PYRO", "AERO", "COSMO", "NOOS"]


class SixLineAttention(nn.Module):
    """
    Multi-head attention where each of the 6 heads is dedicated to one
    hexagram line (domain). Head i learns to attend to tokens whose Q6
    representation has line i activated.

    The hexagram anchor for domain i biases head i's attention scores,
    so each head naturally specialises in one semantic domain.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % 6 == 0, "d_model must be divisible by 6 for SixLineAttention"
        self.n_heads = 6
        self.head_dim = d_model // 6
        self.d_model = d_model
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        hexagrams = get_hexagrams()
        # Per-head domain anchor: project anchor hexagram to head_dim
        self.anchor_proj = nn.ModuleList([
            nn.Linear(6, self.head_dim, bias=False) for _ in range(6)
        ])
        # Register anchor vectors (one per domain/line)
        anchors = []
        for name in DOMAIN_NAMES:
            idx = _DOMAIN_ANCHORS[name]
            anchors.append(hexagrams[idx])  # (6,)
        self.register_buffer("anchors", torch.stack(anchors))  # (6, 6)
        self.domain_lambda = nn.Parameter(torch.ones(6) * 0.1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """x: (B, T, d_model) → (B, T, d_model)."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, 6, self.head_dim).transpose(1, 2)  # (B,6,T,hd)
        k = self.k_proj(x).view(B, T, 6, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, 6, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,6,T,T)

        # Domain anchor bias: project each anchor to head_dim, compute similarity
        for h in range(6):
            anchor_h = self.anchor_proj[h](self.anchors[h])  # (head_dim,)
            # q for head h: (B, T, head_dim) @ anchor_h: (head_dim,) → (B, T)
            q_h = q[:, h, :, :]  # (B, T, head_dim)
            anchor_sim = (q_h * anchor_h.unsqueeze(0).unsqueeze(0)).sum(-1)  # (B, T)
            scores[:, h] = scores[:, h] + self.domain_lambda[h] * anchor_sim.unsqueeze(-1)

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, 6, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Idea 3 — BianGuaOptimizer
# ---------------------------------------------------------------------------

class BianGuaOptimizer(Optimizer):
    """
    SGD-style optimizer that perturbs parameter updates by a Q6-aligned noise
    vector during warmup, encouraging exploration of the weight manifold
    along biangua directions.

    Q6-step: a random {-1,+1}^6 vector (one hexagram vertex) is sampled each
    step and blended with the gradient direction with weight `hex_scale`.
    `hex_scale` follows the same schedule as `AdaptiveHammingScheduler`.

    After warmup, hex_scale anneals to 0 (pure SGD behaviour).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        hex_scale: float = 0.01,
        warmup_steps: int = 100,
        anneal_steps: int = 200,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            hex_scale=hex_scale,
            warmup_steps=warmup_steps,
            anneal_steps=anneal_steps,
        )
        super().__init__(params, defaults)
        self._step_count = 0
        hexagrams = get_hexagrams()
        self.register_buffer = lambda *a: None  # optimizers have no buffers
        self._hexagrams = hexagrams  # (64, 6)

    def _get_hex_scale(self, group: dict) -> float:
        w = group["warmup_steps"]
        a = group["anneal_steps"]
        t = self._step_count
        base = group["hex_scale"]
        if t < w:
            return base * (t / max(w, 1))
        elif t < w + a:
            progress = (t - w) / max(a, 1)
            return base * (1.0 - progress)
        return 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            hs = self._get_hex_scale(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buf"] = torch.zeros_like(p)

                buf = state["momentum_buf"]
                buf.mul_(group["momentum"]).add_(grad)

                if hs > 0.0:
                    # Sample a random hexagram vertex, broadcast to param shape
                    hex_idx = torch.randint(0, 64, (1,)).item()
                    hex_vec = self._hexagrams[hex_idx]  # (6,)
                    # Project hex_vec onto flat param: use first (numel % 6 == 0) elements
                    n = p.numel()
                    rep = (n + 5) // 6
                    hex_noise = hex_vec.repeat(rep)[:n].reshape_as(p)
                    buf.add_(hex_noise, alpha=hs)

                p.add_(buf, alpha=-group["lr"])

        return loss


# ---------------------------------------------------------------------------
# Idea 4 — TernaryKVCache
# ---------------------------------------------------------------------------

class TernaryKVCache(nn.Module):
    """
    KV-cache that compresses keys to {-1, 0, +1} trits (≈1.58 bits/value).

    During encoding (write), keys are quantised via a learned threshold.
    During decoding (read), ternary keys are dequantised by scaling and
    used for approximate attention.

    This halves memory for the key cache while the value cache remains fp16.
    """

    def __init__(self, n_heads: int, head_dim: int, max_seq: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq = max_seq

        # Learned per-head quantisation threshold
        self.log_threshold = nn.Parameter(torch.zeros(n_heads))
        # Cache buffers (allocated lazily)
        self._k_ternary: Optional[Tensor] = None  # (B, n_heads, S, head_dim) int8
        self._v_cache: Optional[Tensor] = None    # (B, n_heads, S, head_dim) float16
        self._cache_len = 0

    @property
    def threshold(self) -> Tensor:
        return F.softplus(self.log_threshold)  # positive, per head

    def _ensure_cache(self, batch: int, device):
        if self._k_ternary is None:
            self._k_ternary = torch.zeros(
                batch, self.n_heads, self.max_seq, self.head_dim,
                dtype=torch.int8, device=device,
            )
            self._v_cache = torch.zeros(
                batch, self.n_heads, self.max_seq, self.head_dim,
                dtype=torch.float16, device=device,
            )
            self._cache_len = 0

    def quantise_keys(self, k: Tensor) -> Tensor:
        """k: (B, H, T, D) float → int8 ternary {-1,0,+1}."""
        thresh = self.threshold.view(1, -1, 1, 1)  # (1, H, 1, 1)
        ternary = torch.zeros_like(k, dtype=torch.int8)
        ternary[k > thresh] = 1
        ternary[k < -thresh] = -1
        return ternary

    def dequantise_keys(self, k_ternary: Tensor) -> Tensor:
        """k_ternary: int8 → float approx."""
        scale = self.threshold.view(1, -1, 1, 1)
        return k_ternary.float() * scale

    def write(self, k: Tensor, v: Tensor):
        """Append keys & values. k, v: (B, H, T, D)."""
        B, H, T, D = k.shape
        self._ensure_cache(B, k.device)
        end = self._cache_len + T
        assert end <= self.max_seq, "Cache overflow"
        self._k_ternary[:, :, self._cache_len:end] = self.quantise_keys(k)
        self._v_cache[:, :, self._cache_len:end] = v.to(torch.float16)
        self._cache_len = end

    def attend(self, q: Tensor) -> Tensor:
        """q: (B, H, T_q, D) → (B, H, T_q, D) attention output."""
        k_approx = self.dequantise_keys(
            self._k_ternary[:, :, :self._cache_len]
        )  # (B, H, S, D)
        v = self._v_cache[:, :, :self._cache_len].float()  # (B, H, S, D)
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k_approx.transpose(-2, -1)) * scale  # (B,H,T_q,S)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def reset(self):
        self._k_ternary = None
        self._v_cache = None
        self._cache_len = 0


# ---------------------------------------------------------------------------
# Idea 5 — HexagramTokenizer
# ---------------------------------------------------------------------------

class HexagramTokenizer:
    """
    64-token vocabulary where each token is one hexagram (a {-1,+1}^6 vertex).
    Text → byte sequence → groups of 6 bits → hexagram index.

    Biangua merge rules (analogous to BPE): two adjacent hexagram tokens that
    are Hamming-1 neighbours are merge candidates. The most frequent pair is
    merged into a new meta-token (stored in merge_table).

    This is a simplified demo tokenizer; full BPE training is in `train_bpe()`.
    """

    PAD_ID = 0

    def __init__(self):
        self.hexagrams = get_hexagrams()  # (64, 6)
        self.biangua = get_biangua()      # (64, 64) adjacency
        self.merge_table: Dict[Tuple[int, int], int] = {}  # (a,b) → meta_id
        self._next_meta = 64  # extended vocab starts after 64 base tokens

    def _bytes_to_gua_ids(self, text: str) -> List[int]:
        """Convert text to hexagram indices via 6-bit chunking of UTF-8 bytes."""
        raw = text.encode("utf-8")
        # Pad to multiple of 6 bits → groups of 6 bits from byte stream
        bits: List[int] = []
        for byte in raw:
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)
        # Pad to multiple of 6
        while len(bits) % 6 != 0:
            bits.append(0)
        ids = []
        for i in range(0, len(bits), 6):
            chunk = bits[i:i + 6]
            # Convert {0,1}^6 → index into hexagrams ({-1,+1}^6)
            val = tuple(1.0 if b else -1.0 for b in chunk)
            # Find matching hexagram
            hex_list = self.hexagrams.tolist()
            idx = hex_list.index(list(val)) if list(val) in hex_list else 0
            ids.append(idx)
        return ids

    def encode(self, text: str) -> List[int]:
        ids = self._bytes_to_gua_ids(text)
        # Apply merge rules (BPE-style)
        changed = True
        while changed:
            changed = False
            new_ids = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids):
                    pair = (ids[i], ids[i + 1])
                    if pair in self.merge_table:
                        new_ids.append(self.merge_table[pair])
                        i += 2
                        changed = True
                        continue
                new_ids.append(ids[i])
                i += 1
            ids = new_ids
        return ids

    def train_bpe(self, corpus: List[str], n_merges: int = 32):
        """Learn biangua-merge rules from corpus."""
        all_ids = [self._bytes_to_gua_ids(t) for t in corpus]
        # Only merge biangua-adjacent pairs (Hamming-1)
        for _ in range(n_merges):
            pair_counts: Dict[Tuple[int, int], int] = {}
            for seq in all_ids:
                for a, b in zip(seq[:-1], seq[1:]):
                    # Only count base pairs (< 64) that are biangua-adjacent
                    if a < 64 and b < 64 and self.biangua[a, b] > 0.5:
                        pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.__getitem__)
            meta_id = self._next_meta
            self.merge_table[best] = meta_id
            self._next_meta += 1
            # Apply merge to all sequences
            for i, seq in enumerate(all_ids):
                new_seq = []
                j = 0
                while j < len(seq):
                    if j + 1 < len(seq) and (seq[j], seq[j + 1]) == best:
                        new_seq.append(meta_id)
                        j += 2
                    else:
                        new_seq.append(seq[j])
                        j += 1
                all_ids[i] = new_seq

    @property
    def vocab_size(self) -> int:
        return self._next_meta


# ---------------------------------------------------------------------------
# Idea 6 — CrossDomainRAG
# ---------------------------------------------------------------------------

class CrossDomainRAG(nn.Module):
    """
    Retrieval-Augmented Generation using Q6 hexagram signatures.

    Documents are indexed by their dominant hexagram (argmax of mean hex_weights).
    At query time, the query's hex_weights are used to retrieve the K most
    similar documents by cosine similarity in the 64-dim hexagram space.

    The retrieved document embeddings are then cross-attended with the query.
    """

    def __init__(self, d_model: int, n_docs: int = 256, top_k: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_docs = n_docs
        self.top_k = top_k

        # Document key embeddings (learned or computed from doc hex_weights)
        self.doc_keys = nn.Parameter(torch.randn(n_docs, 64) * 0.1)  # (N, 64)
        self.doc_values = nn.Parameter(torch.randn(n_docs, d_model) * 0.1)  # (N, d)

        self.q_proj = nn.Linear(64, 64, bias=False)  # query hex → retrieval key
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.doc_embed = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def retrieve(self, query_hex: Tensor) -> Tuple[Tensor, Tensor]:
        """
        query_hex: (B, T, 64) mean hex_weights
        Returns: top_k doc indices, doc similarities
        """
        q = self.q_proj(query_hex)  # (B, T, 64)
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(self.doc_keys, dim=-1)  # (N, 64)
        sim = q_norm @ k_norm.T  # (B, T, N)
        top_scores, top_idx = sim.topk(self.top_k, dim=-1)  # (B, T, K)
        return top_idx, top_scores

    def forward(self, x: Tensor, hex_weights: Tensor) -> Tensor:
        """
        x: (B, T, d_model)
        hex_weights: (B, T, 64)
        Returns: x enriched with retrieved context
        """
        B, T, _ = x.shape
        top_idx, top_scores = self.retrieve(hex_weights)  # (B, T, K)

        # Gather doc values
        idx_flat = top_idx.view(B * T * self.top_k)  # (B*T*K,)
        doc_vals = self.doc_values[idx_flat].view(B * T, self.top_k, self.d_model)

        # Attention weights over top-k docs
        attn_w = F.softmax(top_scores.view(B * T, 1, self.top_k), dim=-1)
        retrieved = (attn_w @ doc_vals).view(B, T, self.d_model)  # (B, T, d)

        # Cross-attend x with retrieved docs
        doc_context = self.doc_embed(retrieved)
        out, _ = self.cross_attn(x.view(B * T, 1, -1), doc_context.view(B * T, 1, -1),
                                  doc_context.view(B * T, 1, -1))
        out = out.view(B, T, self.d_model)
        return x + torch.sigmoid(self.gate) * out


# ---------------------------------------------------------------------------
# Idea 7 — HexagramEvaluator
# ---------------------------------------------------------------------------

class HexagramEvaluator:
    """
    Computes interpretability metrics over Q6 representations.

    Metrics:
      hex_entropy         — Shannon entropy of hex_weights distribution
      domain_coherence    — how concentrated domain weights are (1 = single domain)
      biangua_coverage    — fraction of the 64 hexagrams with weight > threshold
      hamming_entropy_ratio — ratio of hex_entropy to max-entropy (log2(64)=6)
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        biangua = get_biangua()
        self.register_biangua = biangua

    def hex_entropy(self, hex_weights: Tensor) -> Tensor:
        """hex_weights: (B, T, 64) → (B, T) Shannon entropy in bits."""
        eps = 1e-8
        p = hex_weights.clamp(min=eps)
        return -(p * p.log2()).sum(dim=-1)

    def domain_coherence(self, domain_weights: Tensor) -> Tensor:
        """domain_weights: (B, 6) or (B, T, 6) → coherence scalar in [0,1]."""
        p = domain_weights.softmax(dim=-1)
        # Gini-like: 1 means all mass on one domain
        max_p = p.max(dim=-1).values
        return max_p  # (B,) or (B, T)

    def biangua_coverage(self, hex_weights: Tensor) -> Tensor:
        """Fraction of 64 hexagrams with weight > threshold. (B, T) → (B, T)."""
        active = (hex_weights > self.threshold).float()
        return active.mean(dim=-1)  # (B, T)

    def hamming_entropy_ratio(self, hex_weights: Tensor) -> Tensor:
        """Normalised entropy: 0=collapsed, 1=uniform over Q6."""
        max_entropy = math.log2(64)  # 6 bits
        return self.hex_entropy(hex_weights) / max_entropy

    def evaluate(self, hex_weights: Tensor, domain_weights: Optional[Tensor] = None) -> dict:
        """Full evaluation dict."""
        result = {
            "hex_entropy": self.hex_entropy(hex_weights).mean().item(),
            "biangua_coverage": self.biangua_coverage(hex_weights).mean().item(),
            "hamming_entropy_ratio": self.hamming_entropy_ratio(hex_weights).mean().item(),
        }
        if domain_weights is not None:
            result["domain_coherence"] = self.domain_coherence(domain_weights).mean().item()
        return result


# ---------------------------------------------------------------------------
# Idea 8 — MultiScaleQ6 (Matryoshka hierarchy)
# ---------------------------------------------------------------------------

class MultiScaleQ6(nn.Module):
    """
    Matryoshka representation over Q6 sub-hypercubes.

    Hierarchy:
      Q2 — {-1,+1}^2 = 4 vertices  (first 2 lines)
      Q3 — {-1,+1}^3 = 8 vertices  (first 3 lines)
      Q6 — {-1,+1}^6 = 64 vertices (all 6 lines)

    Each scale produces a soft assignment and a projected embedding.
    The final output is a weighted sum across scales.
    """

    def __init__(self, d_model: int, temperature: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

        # Build sub-hypercubes
        def make_sub(dims: int) -> Tensor:
            verts = list(itertools.product([-1.0, 1.0], repeat=dims))
            return torch.tensor(verts, dtype=torch.float32)

        self.register_buffer("q2_verts", make_sub(2))   # (4, 2)
        self.register_buffer("q3_verts", make_sub(3))   # (8, 3)
        self.register_buffer("q6_verts", get_hexagrams())  # (64, 6)

        # Projections: d_model → k dimensions → softmax → weighted vertex
        self.proj_q2 = nn.Linear(d_model, 2, bias=False)
        self.proj_q3 = nn.Linear(d_model, 3, bias=False)
        self.proj_q6 = nn.Linear(d_model, 6, bias=False)

        # Scale-to-d_model projections
        self.out_q2 = nn.Linear(2, d_model, bias=False)
        self.out_q3 = nn.Linear(3, d_model, bias=False)
        self.out_q6 = nn.Linear(6, d_model, bias=False)

        # Scale mixing weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3.0)

    def _soft_embed(self, h: Tensor, proj: nn.Linear, verts: Tensor) -> Tuple[Tensor, Tensor]:
        """h: (B,T,d) → soft_embed:(B,T,d), weights:(B,T,n_verts)."""
        soft = torch.tanh(proj(h))  # (B, T, dims)
        sim = soft @ verts.T / self.temperature  # (B, T, n_verts)
        weights = F.softmax(sim, dim=-1)
        embed = weights @ verts  # (B, T, dims)
        return embed, weights

    def forward(self, x: Tensor) -> Tuple[Tensor, dict]:
        """x: (B, T, d_model) → enriched: (B, T, d_model), scale_info: dict."""
        e2, w2 = self._soft_embed(x, self.proj_q2, self.q2_verts)
        e3, w3 = self._soft_embed(x, self.proj_q3, self.q3_verts)
        e6, w6 = self._soft_embed(x, self.proj_q6, self.q6_verts)

        scale_w = F.softmax(self.scale_weights, dim=0)
        combined = (
            scale_w[0] * self.out_q2(e2)
            + scale_w[1] * self.out_q3(e3)
            + scale_w[2] * self.out_q6(e6)
        )
        return x + combined, {"q2_weights": w2, "q3_weights": w3, "q6_weights": w6}


# ---------------------------------------------------------------------------
# Idea 9 — AdaptiveHammingScheduler
# ---------------------------------------------------------------------------

class AdaptiveHammingScheduler:
    """
    Curriculum schedule for the BianGuaAttention hamming_lambda parameter.

    Schedule (in steps):
      [0, warmup)         — linear ramp 0 → lambda_max
      [warmup, warmup+anneal) — cosine decay lambda_max → lambda_min
      [warmup+anneal, ...)    — steady at lambda_min

    Usage:
        scheduler = AdaptiveHammingScheduler(model, warmup=500, anneal=1000)
        for step in training_loop:
            scheduler.step(step)
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_max: float = 0.5,
        lambda_min: float = 0.05,
        warmup_steps: int = 500,
        anneal_steps: int = 1000,
    ):
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.warmup_steps = warmup_steps
        self.anneal_steps = anneal_steps
        # Collect all BianGuaAttention modules in the model
        from yijing_transformer.models.variant3 import BianGuaAttention
        self.targets = [m for m in model.modules() if isinstance(m, BianGuaAttention)]

    def get_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lambda_max * (step / max(self.warmup_steps, 1))
        t = step - self.warmup_steps
        if t < self.anneal_steps:
            progress = t / max(self.anneal_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.lambda_min + (self.lambda_max - self.lambda_min) * cosine
        return self.lambda_min

    def step(self, step: int):
        lam = self.get_lambda(step)
        for module in self.targets:
            module.hamming_lambda.data.fill_(lam)


# ---------------------------------------------------------------------------
# Idea 10 — HexagramMoE
# ---------------------------------------------------------------------------

class HexagramMoE(nn.Module):
    """
    Mixture-of-Experts with 64 experts, one per hexagram vertex.

    Routing: hex_weights (B, T, 64) directly selects expert mixture.
    Top-K experts are selected; their outputs are combined by weight.

    Vectorised implementation: experts share W_in/W_out tensor banks.
    Expert i computes: FFN_i(x) = W_out[i] * SiLU(W_in[i] * x)
    """

    def __init__(self, d_model: int, d_ff: int, top_k: int = 4):
        super().__init__()
        self.n_experts = 64
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k

        # Vectorised expert banks: (64, d_ff, d_model) and (64, d_model, d_ff)
        self.W_in = nn.Parameter(torch.randn(64, d_ff, d_model) * (d_model ** -0.5))
        self.W_out = nn.Parameter(torch.randn(64, d_model, d_ff) * (d_ff ** -0.5))
        self.bias_in = nn.Parameter(torch.zeros(64, d_ff))
        self.bias_out = nn.Parameter(torch.zeros(64, d_model))

        # Load balancing: track expert utilisation
        self.register_buffer("expert_counts", torch.zeros(64))

    def forward(self, x: Tensor, hex_weights: Tensor) -> Tensor:
        """
        x: (B, T, d_model)
        hex_weights: (B, T, 64) routing weights
        Returns: (B, T, d_model) MoE output
        """
        B, T, D = x.shape

        # Top-K routing
        top_w, top_idx = hex_weights.topk(self.top_k, dim=-1)  # (B, T, K)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-8)  # renorm

        # Update utilisation counts (for load balancing loss)
        if self.training:
            with torch.no_grad():
                counts = torch.zeros(64, device=x.device)
                counts.scatter_add_(0, top_idx.reshape(-1),
                                     torch.ones(B * T * self.top_k, device=x.device))
                self.expert_counts.mul_(0.99).add_(counts * 0.01)

        # Compute expert outputs for selected experts
        # x_flat: (B*T, D)
        x_flat = x.view(B * T, D)
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            idx_k = top_idx[:, :, k].reshape(B * T)   # (B*T,)
            w_k = top_w[:, :, k].reshape(B * T, 1)    # (B*T, 1)

            # Gather expert weights for each token: bmm approach
            W_in_k = self.W_in[idx_k]   # (B*T, d_ff, D)
            b_in_k = self.bias_in[idx_k]  # (B*T, d_ff)
            W_out_k = self.W_out[idx_k]  # (B*T, D, d_ff)
            b_out_k = self.bias_out[idx_k]  # (B*T, D)

            h = torch.bmm(W_in_k, x_flat.unsqueeze(-1)).squeeze(-1) + b_in_k  # (B*T, d_ff)
            h = F.silu(h)
            out_k = torch.bmm(W_out_k, h.unsqueeze(-1)).squeeze(-1) + b_out_k  # (B*T, D)
            output += w_k * out_k

        return output.view(B, T, D)

    def load_balance_loss(self, hex_weights: Tensor) -> Tensor:
        """Auxiliary loss to encourage uniform expert utilisation."""
        mean_routing = hex_weights.mean(dim=[0, 1])  # (64,)
        return (mean_routing * self.expert_counts).sum()


# ===========================================================================
# TEXT-ANALYSIS MODULES
# (derived from the antonym/conveyor/Forth/own-alien analysis)
# ===========================================================================

# ---------------------------------------------------------------------------
# BinaryOppositionTable — antonym pairs as Q6 axes
# ---------------------------------------------------------------------------

@dataclass
class AxisDefinition:
    """One binary opposition axis."""
    name: str
    positive: str   # +1 pole
    negative: str   # -1 pole
    domain: str     # which of 6 domains this axis belongs to


DEFAULT_AXES: List[AxisDefinition] = [
    AxisDefinition("temporality",   "permanent",   "transient",   "COSMO"),
    AxisDefinition("proximity",     "near",        "far",         "GEO"),
    AxisDefinition("energy",        "active",      "passive",     "PYRO"),
    AxisDefinition("order",         "structured",  "chaotic",     "NOOS"),
    AxisDefinition("materiality",   "concrete",    "abstract",    "GEO"),
    AxisDefinition("polarity",      "positive",    "negative",    "AERO"),
]


class BinaryOppositionTable(nn.Module):
    """
    Maps arbitrary text features to the 6 Q6 binary axes.

    Each axis corresponds to one hexagram line. A feature vector x is
    projected to 6 axis scores ∈ [-1, +1], producing a soft hexagram
    coordinate. This coordinate can then be looked up in the Q6 vocabulary.

    The table stores learned projections from d_model → 6 (one per axis).
    """

    def __init__(self, d_model: int, axes: Optional[List[AxisDefinition]] = None):
        super().__init__()
        self.axes = axes or DEFAULT_AXES
        assert len(self.axes) == 6, "Must have exactly 6 axes for Q6"

        self.proj = nn.Linear(d_model, 6, bias=True)
        self.register_buffer("hexagrams", get_hexagrams())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: (B, T, d_model)
        Returns:
          axis_scores: (B, T, 6) — soft coordinate in [-1,+1]^6
          hex_assignment: (B, T, 64) — softmax similarity to all hexagrams
        """
        axis_scores = torch.tanh(self.proj(x))  # (B, T, 6)
        sim = axis_scores @ self.hexagrams.T    # (B, T, 64)
        hex_assignment = F.softmax(sim / 0.5, dim=-1)
        return axis_scores, hex_assignment

    def interpret(self, axis_scores: Tensor) -> List[List[str]]:
        """Convert axis_scores to human-readable pole labels."""
        results = []
        for score in axis_scores.view(-1, 6).tolist():
            row = []
            for ax, s in zip(self.axes, score):
                label = ax.positive if s > 0 else ax.negative
                row.append(f"{ax.name}={label}({s:+.2f})")
            results.append(row)
        return results


# ---------------------------------------------------------------------------
# SvoyChuzhoiGate — own/alien classification via Q6 distance
# ---------------------------------------------------------------------------

class SvoyChuzhoiGate(nn.Module):
    """
    "Свой/Чужой" (own/alien) gate using Q6 distance to a prototype.

    Derived from the 'свой/чужой' concept in the analysis text:
    - 'Свой' (own, familiar): close to prototype in Q6 space → gate = +1
    - 'Нейтральный' (neutral, uncertain): mid-distance → gate = 0
    - 'Чужой' (alien, unfamiliar): far from prototype → gate = -1

    This is a semantic analogue of TernaryGate but based on Q6 topology
    rather than learned thresholds.

    The prototype is a learnable hexagram coordinate.
    """

    def __init__(self, d_model: int, n_prototypes: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_prototypes = n_prototypes

        # Learnable prototype coordinates in Q6 space
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, 6) * 0.5
        )  # (P, 6)
        # Project input to Q6 space
        self.proj = nn.Linear(d_model, 6, bias=False)
        # How to map distance to gate
        self.near_threshold = nn.Parameter(torch.tensor(1.5))   # <1.5 → own
        self.far_threshold = nn.Parameter(torch.tensor(3.5))    # >3.5 → alien
        # Output scale
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: (B, T, d_model)
        Returns:
          gated: (B, T, d_model) — x modulated by svoy/chuzhoi gate
          gate_values: (B, T) — gate in {-1, 0, +1} (STE)
        """
        q6_x = torch.tanh(self.proj(x))  # (B, T, 6) — soft coord
        proto_norm = torch.tanh(self.prototypes)  # (P, 6)

        # Minimum Q6 distance to any prototype (using L2 in continuous embedding)
        dists = torch.cdist(
            q6_x.view(-1, 6),
            proto_norm
        ).view(x.shape[0], x.shape[1], self.n_prototypes)  # (B, T, P)
        min_dist = dists.min(dim=-1).values  # (B, T)

        # Soft gate based on distance
        near_thresh = F.softplus(self.near_threshold)
        far_thresh = F.softplus(self.far_threshold) + near_thresh

        # Continuous version
        gate_soft = torch.zeros_like(min_dist)
        gate_soft = gate_soft + torch.sigmoid(near_thresh - min_dist)  # near → +1
        gate_soft = gate_soft - torch.sigmoid(min_dist - far_thresh)   # far → -1

        # Hard quantisation with STE
        gate_hard = torch.zeros_like(gate_soft)
        gate_hard[min_dist < near_thresh] = 1.0
        gate_hard[min_dist > far_thresh] = -1.0
        gate_ste = gate_soft + (gate_hard - gate_soft).detach()

        gated = self.out_proj(x * gate_ste.unsqueeze(-1))
        return gated, gate_ste


# ---------------------------------------------------------------------------
# BinaryExclusionClassifier — method of exclusion via per-axis AND
# ---------------------------------------------------------------------------

class BinaryExclusionClassifier(nn.Module):
    """
    Method-of-Exclusion classifier using the binary opposition table.

    For each of 6 axes, a binary accept/reject decision is made. A sample
    passes (is "own") only if it passes ALL 6 axis checks — this is the
    AND combination (exclusion by failure on any axis).

    Analogous to bicycle route safety: a path is safe only if ALL 6 safety
    criteria are satisfied (each line of the hexagram must be valid).

    The 6 classifiers use the axis projections from BinaryOppositionTable.
    """

    def __init__(
        self,
        d_model: int,
        axes: Optional[List[AxisDefinition]] = None,
        threshold: float = 0.0,
    ):
        super().__init__()
        self.threshold = threshold
        self.opp_table = BinaryOppositionTable(d_model, axes)
        # Learned per-axis threshold offsets
        self.axis_thresholds = nn.Parameter(torch.zeros(6))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        x: (B, T, d_model)
        Returns:
          accept_mask: (B, T) bool — True if passes ALL 6 axis checks
          axis_scores: (B, T, 6) — per-axis soft scores
          axis_decisions: (B, T, 6) bool — per-axis binary decisions
        """
        axis_scores, hex_assignment = self.opp_table(x)
        thresh = self.axis_thresholds.unsqueeze(0).unsqueeze(0)  # (1,1,6)
        axis_decisions = axis_scores > thresh  # (B, T, 6) bool

        # AND: accept only if ALL axes pass
        accept_mask = axis_decisions.all(dim=-1)  # (B, T)

        return accept_mask, axis_scores, axis_decisions


# ---------------------------------------------------------------------------
# TextQualityFilter — 6-axis quality hexagram
# ---------------------------------------------------------------------------

class TextQualityFilter(nn.Module):
    """
    Text quality assessment using a 6-bit hexagram score.

    Each of the 6 quality dimensions maps to one hexagram line:
      Line 1 (GEO): is_factual       — factual/grounded vs. vague/hallucinated
      Line 2 (HYDRO): is_coherent    — coherent flow vs. contradictory
      Line 3 (PYRO): is_relevant     — on-topic vs. off-topic
      Line 4 (AERO): is_clear        — clear language vs. ambiguous
      Line 5 (COSMO): is_complete    — complete answer vs. partial
      Line 6 (NOOS): is_safe         — safe/benign content vs. harmful

    Score 63 (111111 in binary) = "hexagram 64" (all lines positive) = high quality
    Score 0  (000000 in binary) = "hexagram 1" (all lines negative) = spam/low quality

    The filter is a learned classifier that can be used as a reward signal.
    """

    QUALITY_AXES = [
        "is_factual", "is_coherent", "is_relevant",
        "is_clear", "is_complete", "is_safe"
    ]

    def __init__(self, d_model: int):
        super().__init__()
        # Pool sequence → single vector, then classify on 6 axes
        self.pool = nn.Linear(d_model, d_model, bias=False)
        self.quality_heads = nn.Linear(d_model, 6, bias=True)
        self.register_buffer("hexagrams", get_hexagrams())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, int]:
        """
        x: (B, T, d_model)
        Returns:
          quality_scores: (B, 6) — per-axis quality in [-1, +1]
          hex_assignment: (B, 64) — closest hexagram
          dominant_idx: int — batch-mean dominant hexagram index
        """
        # Mean-pool over sequence
        pooled = self.pool(x.mean(dim=1))  # (B, d_model)
        quality_scores = torch.tanh(self.quality_heads(pooled))  # (B, 6)

        # Map to hexagram
        sim = quality_scores @ self.hexagrams.T  # (B, 64)
        hex_assignment = F.softmax(sim / 0.5, dim=-1)

        dominant_idx = hex_assignment.mean(dim=0).argmax().item()

        return quality_scores, hex_assignment, dominant_idx

    def quality_bits(self, quality_scores: Tensor) -> Tensor:
        """Convert quality scores to binary hexagram index (0–63)."""
        bits = (quality_scores > 0.0).long()  # (B, 6)
        powers = torch.tensor([32, 16, 8, 4, 2, 1], device=bits.device)
        return (bits * powers).sum(dim=-1)  # (B,) in [0, 63]


# ---------------------------------------------------------------------------
# ConveyorVariant3Block — named 6-stage pipeline with inspection
# ---------------------------------------------------------------------------

class ConveyorStageOutput:
    """Named outputs from each conveyor stage for inspection."""
    def __init__(self):
        self.stages: Dict[str, Tensor] = {}
        self.hex_weights: Optional[Tensor] = None

    def record(self, name: str, tensor: Tensor):
        self.stages[name] = tensor.detach()

    def __repr__(self) -> str:
        shapes = {k: tuple(v.shape) for k, v in self.stages.items()}
        return f"ConveyorStageOutput({shapes})"


class ConveyorVariant3Block(nn.Module):
    """
    Named 6-stage conveyor version of Variant3Block.

    The 6 stages (conveyor positions) are:
      1. Q6_LOCALISE   — HexagramProjection: locate token in Q6 hypercube
      2. TOPO_ATTEND   — BianGuaAttention: topological self-attention
      3. TERNARY_GATE  — TernaryGate: filter {-1,0,+1}
      4. INTERLINGUA   — ArchetypalInterlingua: hub-and-spoke mediation
      5. BIANGUA_ANALOGY — CrossHexagramAnalogy: 変爻 reasoning
      6. SWIGLU_FFN    — SwiGLU feed-forward: nonlinear synthesis

    With `record_intermediates=True`, all intermediate activations are
    captured in a ConveyorStageOutput object for interpretability.
    """

    STAGE_NAMES = [
        "Q6_LOCALISE",
        "TOPO_ATTEND",
        "TERNARY_GATE",
        "INTERLINGUA",
        "BIANGUA_ANALOGY",
        "SWIGLU_FFN",
    ]

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4,
                 hamming_lambda: float = 0.1, uncertainty_budget: float = 0.3,
                 dropout: float = 0.0):
        super().__init__()
        from yijing_transformer.models.variant3 import (
            HexagramProjection, BianGuaAttention, TernaryGate,
            CrossHexagramAnalogy
        )
        from yijing_transformer.models.geometry.routing import ArchetypalInterlingua

        self.norm_hex = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        self.hex_proj = HexagramProjection(d_model)
        self.biangua_attn = BianGuaAttention(d_model, n_heads, hamming_lambda)
        self.ternary_gate = TernaryGate(d_model, uncertainty_budget)
        self.interlingua = ArchetypalInterlingua(d_model, n_sources=2)
        self.analogy = CrossHexagramAnalogy(d_model)

        d_ff = d_model * ffn_mult
        self.ffn_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_value_proj = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_out_proj = nn.Linear(d_ff, d_model, bias=False)
        self.ffn_drop = nn.Dropout(dropout)

        self.record_intermediates = False
        self._last_output: Optional[ConveyorStageOutput] = None

    def _ffn(self, x: Tensor) -> Tensor:
        gate = F.silu(self.ffn_gate_proj(x))
        val = self.ffn_value_proj(x)
        return self.ffn_out_proj(self.ffn_drop(gate * val))

    def forward(self, x: Tensor) -> Tensor:
        out = ConveyorStageOutput() if self.record_intermediates else None

        # Stage 1: Q6_LOCALISE
        h_enriched, hex_weights = self.hex_proj(self.norm_hex(x))
        x = x + (h_enriched - self.norm_hex(x))
        if out:
            out.record("Q6_LOCALISE", x)
            out.hex_weights = hex_weights.detach()

        # Stage 2: TOPO_ATTEND
        x = x + self.biangua_attn(self.norm_attn(x), hex_weights)
        if out:
            out.record("TOPO_ATTEND", x)

        # Stage 3: TERNARY_GATE
        attn_out = x
        x = self.ternary_gate(x)
        ternary_out = x
        if out:
            out.record("TERNARY_GATE", x)

        # Stage 4: INTERLINGUA
        x = self.interlingua(attn_out, [attn_out, ternary_out])
        if out:
            out.record("INTERLINGUA", x)

        # Stage 5: BIANGUA_ANALOGY
        x = self.analogy(x, hex_weights)
        if out:
            out.record("BIANGUA_ANALOGY", x)

        # Stage 6: SWIGLU_FFN
        x = x + self._ffn(self.norm_ffn(x))
        if out:
            out.record("SWIGLU_FFN", x)
            self._last_output = out

        return x

    @property
    def last_stage_output(self) -> Optional[ConveyorStageOutput]:
        """Retrieve the last recorded intermediate outputs."""
        return self._last_output

#!/usr/bin/env python3
"""
bidir_train.py — Замкнутый двунаправленный цикл обучения.

ТЕОРИЯ (data7/knowledge_transformation_theory.md):
  Диссертации ⇄ Энциклопедии
  K₀ → decompose → aggregate → K₁ → decompose → synthesize → K₂ → ...

НЕДОСТАЮЩАЯ ПЕТЛЯ (data7/knowledge_transformer.py, строка-комментарий):
  # Missing: proposals → decomposer.decompose() → refinement loop

ЗДЕСЬ ЭТА ПЕТЛЯ РЕАЛИЗОВАНА:

  ┌─────────────────────────────────────────────────────────────────┐
  │             ДВУНАПРАВЛЕННЫЙ ЦИКЛ                                │
  │                                                                 │
  │  ВПЕРЁД (специализация → обобщение):                           │
  │  Корпус → KnowledgeGraph → PageRank-центры → Q6-анкоры         │
  │  → Variant3GPT учится на центральных концептах                  │
  │                                                                 │
  │  НАЗАД (обобщение → специализация):                             │
  │  Variant3GPT генерирует → QFilter оценивает                     │
  │  → AdaptiveLearning обновляет веса рёбер графа                  │
  │  → identify_gaps() находит новые пробелы                        │
  │  → generate_hypotheses() → новый корпус → снова вперёд          │
  │                                                                 │
  │  КРИТЕРИЙ ЗАВЕРШЕНИЯ: модель сама генерирует тексты,            │
  │  которые граф знаний признаёт "достаточно центральными"         │
  └─────────────────────────────────────────────────────────────────┘

АНАЛОГИЯ С data7:
  KnowledgeGraph.compute_centrality()  ↔  hex_weights (мягкое внимание)
  identify_gaps()                       ↔  domain_triplet_loss (разрыв)
  generate_hypotheses()                 ↔  self_dialog stage 3
  AdaptiveLearningOptimizer             ↔  gradient descent на QFilter
  TSP-оптимизация порядка               ↔  BFS-путь по biangua-графу
"""

import os, sys, math, random, collections, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from yijing_transformer.models.variant3 import (
    Variant3Config, Variant3GPT,
    DOMAINS, DOMAIN_ANCHORS,
)
from yijing_transformer.models.variant3_extensions import (
    HexagramEvaluator, TextQualityFilter,
    get_hexagrams, get_biangua, bfs_distances,
)
from yijing_transformer.constants import HEX_NAMES

# ─── конфигурация ─────────────────────────────────────────────────────────────

torch.manual_seed(42)
random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = Variant3Config(
    vocab_size=256, block_size=32, d_model=128,
    n_heads=4, n_layers=4, ffn_mult=4,
    hamming_lambda=0.15, uncertainty_budget=0.25,
    dropout=0.05, use_domain_routing=True,
)

# Веса лоссов
ALPHA = 0.30   # domain triplet
BETA  = 0.20   # quality contrastive
GAMMA = 0.10   # gate entropy

hexagrams   = get_hexagrams()
biangua     = get_biangua()
dist_matrix = bfs_distances(biangua)
evaluator   = HexagramEvaluator(threshold=0.01)
qfilter     = TextQualityFilter(CFG.d_model)

def hname(i): return HEX_NAMES[i] if i < 64 else f"#{i}"


# ═══════════════════════════════════════════════════════════════════════════
# ЧАСТЬ I: KNOWLEDGE GRAPH (аналог data7/knowledge_transformer.py)
# ═══════════════════════════════════════════════════════════════════════════

class Concept:
    """Атомарная единица знания. Аналог data7: Concept."""
    def __init__(self, name: str, domain: str, depth: float = 0.5,
                 novelty: float = 0.5, certainty: float = 0.8):
        self.name      = name
        self.domain    = domain
        self.depth     = depth     # специализация (0=поверхностно, 1=глубоко)
        self.novelty   = novelty   # новизна (0=известно, 1=новое)
        self.certainty = certainty
        # Q6-позиция (назначается позже по model forward pass)
        self.hex_idx: int = -1
        self.pagerank: float = 0.0


class KnowledgeGraph:
    """
    Граф научных знаний. Аналог data7: KnowledgeGraph.

    Отличие: рёбра взвешены качеством связи (обновляется AdaptiveLearning).
    Узлы = концепты, ребра = смысловые связи.
    """

    RELATION_TYPES = {
        "causes", "extends", "contradicts", "related_to",
        "is_a", "part_of", "applies_to", "derived_from",
    }

    def __init__(self):
        self.concepts: dict[str, Concept] = {}
        # adj[a][b] = {"type": str, "weight": float, "certainty": float}
        self.adj: dict[str, dict[str, dict]] = {}
        self._pagerank_dirty = True

    def add_concept(self, c: Concept):
        self.concepts[c.name] = c
        if c.name not in self.adj:
            self.adj[c.name] = {}
        self._pagerank_dirty = True

    def add_relation(self, src: str, dst: str, rel_type: str,
                     weight: float = 1.0, certainty: float = 0.8):
        if src not in self.adj:
            self.adj[src] = {}
        self.adj[src][dst] = {
            "type": rel_type, "weight": weight, "certainty": certainty
        }
        # bidirectional (слабое обратное ребро)
        if dst not in self.adj:
            self.adj[dst] = {}
        if src not in self.adj[dst]:
            self.adj[dst][src] = {
                "type": rel_type, "weight": weight * 0.5, "certainty": certainty
            }
        self._pagerank_dirty = True

    def compute_centrality(self, damping: float = 0.85, iterations: int = 30):
        """
        PageRank. Аналог data7: compute_centrality().
        Веса рёбер учитываются (weighted PageRank).
        """
        nodes = list(self.concepts.keys())
        if not nodes:
            return
        n = len(nodes)
        rank = {k: 1.0 / n for k in nodes}

        for _ in range(iterations):
            new_rank = {}
            for v in nodes:
                # Сумма входящих рёбер с весами
                in_sum = 0.0
                for u in nodes:
                    if v in self.adj.get(u, {}):
                        edge = self.adj[u][v]
                        # Суммарный исходящий вес из u
                        out_total = sum(
                            e["weight"] for e in self.adj[u].values()
                        ) + 1e-8
                        in_sum += rank[u] * edge["weight"] / out_total
                new_rank[v] = (1 - damping) / n + damping * in_sum
            rank = new_rank

        total = sum(rank.values()) + 1e-8
        for name, c in self.concepts.items():
            c.pagerank = rank.get(name, 0) / total
        self._pagerank_dirty = False

    def identify_gaps(self, top_k: int = 10):
        """
        Находит пробелы в знаниях.
        Аналог data7: identify_knowledge_gaps().

        Пробел = пара концептов, между которыми нет прямого ребра,
        но они достаточно близки по доменной структуре.
        """
        if self._pagerank_dirty:
            self.compute_centrality()

        gaps = []
        nodes = list(self.concepts.keys())
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                if b in self.adj.get(a, {}):
                    continue   # прямая связь есть
                ca = self.concepts[a]
                cb = self.concepts[b]
                # Потенциал = среднее PageRank × похожесть доменов
                domain_sim = 1.0 if ca.domain == cb.domain else 0.3
                potential = (ca.pagerank + cb.pagerank) / 2 * domain_sim
                if potential > 0.005:
                    gaps.append({
                        "a": a, "b": b,
                        "potential": potential,
                        "same_domain": ca.domain == cb.domain,
                    })

        gaps.sort(key=lambda g: g["potential"], reverse=True)
        return gaps[:top_k]

    def generate_hypotheses(self, gaps: list) -> list[str]:
        """
        Генерирует текстовые гипотезы из пробелов.
        Аналог data7: generate_hypotheses().

        Возвращает список текстов-гипотез для обучающего корпуса.
        """
        templates = [
            "{a} may extend our understanding of {b} through shared mechanisms.",
            "The relationship between {a} and {b} reveals structural patterns.",
            "{a} causes or influences {b} in complex systems.",
            "Combining {a} with {b} enables new theoretical frameworks.",
            "{a} and {b} are derived from common foundational principles.",
            "The interaction of {a} and {b} deserves systematic investigation.",
            "{a} applies directly to problems involving {b}.",
            "Understanding {a} is prerequisite to mastering {b}.",
        ]
        hypotheses = []
        for gap in gaps:
            tmpl = random.choice(templates)
            text = tmpl.format(a=gap["a"], b=gap["b"])
            hypotheses.append(text)
        return hypotheses

    def adaptive_update(self, src: str, dst: str, quality_delta: float,
                        lr: float = 0.1):
        """
        AdaptiveLearningOptimizer из data7/advanced_methods.py.

        Обновляет вес ребра на основе качества сгенерированных текстов.
        quality_delta > 0: связь подтверждена моделью → усиливаем ребро
        quality_delta < 0: связь не подтверждена → ослабляем
        """
        if src in self.adj and dst in self.adj[src]:
            old = self.adj[src][dst]["weight"]
            new = old + lr * quality_delta
            self.adj[src][dst]["weight"] = max(0.01, min(5.0, new))
        else:
            # Новое ребро: модель нашла связь которой не было в графе
            if src not in self.adj:
                self.adj[src] = {}
            self.adj[src][dst] = {
                "type": "related_to",
                "weight": max(0.01, 0.5 + lr * quality_delta),
                "certainty": 0.5,
            }
        self._pagerank_dirty = True

    def tsp_order(self, subset: list[str]) -> list[str]:
        """
        TSP-оптимизация порядка изложения концептов.
        Аналог data7: dissertation_optimizer + cognitive_distance.

        Здесь: greedy nearest-neighbor по PageRank-weighted расстоянию.
        """
        if len(subset) <= 1:
            return subset

        # Стартуем с наиболее центрального
        if self._pagerank_dirty:
            self.compute_centrality()
        start = max(subset, key=lambda n: self.concepts[n].pagerank
                    if n in self.concepts else 0)

        visited = [start]
        remaining = [n for n in subset if n != start]

        while remaining:
            cur = visited[-1]
            # "Когнитивное расстояние" = 1/weight если ребро есть, иначе 10
            def cognitive_dist(nxt):
                edge = self.adj.get(cur, {}).get(nxt)
                if edge:
                    return 1.0 / (edge["weight"] + 1e-8)
                return 10.0

            nxt = min(remaining, key=cognitive_dist)
            visited.append(nxt)
            remaining.remove(nxt)

        return visited

    def summary(self) -> dict:
        if self._pagerank_dirty:
            self.compute_centrality()
        top5 = sorted(self.concepts.values(),
                      key=lambda c: c.pagerank, reverse=True)[:5]
        return {
            "nodes": len(self.concepts),
            "edges": sum(len(v) for v in self.adj.values()),
            "top_by_pagerank": [(c.name, f"{c.pagerank:.4f}") for c in top5],
        }


# ═══════════════════════════════════════════════════════════════════════════
# ЧАСТЬ II: НАЧАЛЬНЫЙ КОРПУС (домены → граф → гипотезы)
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_CORPUS = {
    "PYRO":  [
        ("fire",        "Fire transforms matter through rapid oxidation."),
        ("light",       "Photons carry electromagnetic energy as light."),
        ("combustion",  "Combustion converts chemical energy to heat."),
        ("plasma",      "Plasma is ionized gas at extreme temperatures."),
        ("radiation",   "Thermal radiation transfers heat through space."),
    ],
    "HYDRO": [
        ("water",       "Water molecules bond through hydrogen forces."),
        ("flow",        "Rivers erode valleys over geological time."),
        ("ocean",       "Oceans regulate global temperature cycles."),
        ("wave",        "Waves propagate energy across water surfaces."),
        ("pressure",    "Hydraulic pressure transmits force uniformly."),
    ],
    "AERO":  [
        ("wind",        "Wind patterns arise from differential heating."),
        ("air",         "Air pressure gradients drive atmospheric flow."),
        ("turbulence",  "Turbulence emerges from unstable fluid dynamics."),
        ("sound",       "Sound waves are longitudinal pressure variations."),
        ("breath",      "Breathing exchanges gases through membrane diffusion."),
    ],
    "GEO":   [
        ("mountain",    "Mountains form where tectonic plates collide."),
        ("soil",        "Soil composition determines ecosystem productivity."),
        ("earthquake",  "Earthquakes release stress along fault lines."),
        ("crystal",     "Crystals self-organize through atomic bonding."),
        ("erosion",     "Erosion shapes landscapes through particle transport."),
    ],
    "COSMO": [
        ("galaxy",      "Galaxies form from gravitational collapse of gas."),
        ("star",        "Stars synthesize heavy elements through fusion."),
        ("blackhole",   "Black holes warp spacetime beyond escape velocity."),
        ("cosmos",      "The cosmic web connects all visible structure."),
        ("time",        "Time dilation links velocity to temporal flow."),
    ],
    "NOOS":  [
        ("logic",       "Logic derives truth through valid inference rules."),
        ("language",    "Language encodes meaning through symbolic systems."),
        ("proof",       "Mathematical proofs chain axioms to conclusions."),
        ("mind",        "Consciousness integrates distributed neural signals."),
        ("pattern",     "Pattern recognition generalizes from finite examples."),
    ],
}

BAD_TEXTS = [
    "spam click win free prize limited offer now",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "x x x x x x x x x x x x x x x",
    "asdf qwer zxcv random noise garbage input",
    "0000000000000000000000000000000000",
    "buy now click here special deal today only",
]


def build_initial_graph(domain_corpus: dict) -> KnowledgeGraph:
    """
    Строим граф знаний из доменного корпуса.
    Аналог data7: DissertationDecomposer + KnowledgeGraph.

    Концепты = именованные сущности,
    Связи    = внутри домена (все пары) + между доменами (diagonal)
    """
    g = KnowledgeGraph()

    # Добавляем концепты
    for domain, entries in domain_corpus.items():
        for name, _ in entries:
            depth   = random.uniform(0.4, 0.9)
            novelty = random.uniform(0.3, 0.8)
            g.add_concept(Concept(name, domain, depth, novelty))

    # Внутридоменные связи (каждый с каждым)
    for domain, entries in domain_corpus.items():
        names = [n for n, _ in entries]
        for i, a in enumerate(names):
            for b in names[i+1:]:
                rel = random.choice(["extends", "related_to", "applies_to"])
                w = random.uniform(0.5, 1.5)
                g.add_relation(a, b, rel, weight=w)

    # Межсдоменные связи (избранные, высокий потенциал)
    cross_domain_pairs = [
        ("fire",     "light",    "causes",     1.2),
        ("water",    "erosion",  "causes",     1.1),
        ("wind",     "wave",     "causes",     1.0),
        ("mountain", "erosion",  "related_to", 0.9),
        ("star",     "light",    "causes",     1.3),
        ("logic",    "proof",    "is_a",       1.4),
        ("language", "pattern",  "extends",    0.8),
        ("blackhole","time",     "causes",     1.1),
        ("plasma",   "star",     "is_a",       1.2),
        ("crystal",  "pattern",  "related_to", 0.7),
    ]
    for a, b, rel, w in cross_domain_pairs:
        if a in g.concepts and b in g.concepts:
            g.add_relation(a, b, rel, weight=w, certainty=0.85)

    g.compute_centrality()
    return g


# ═══════════════════════════════════════════════════════════════════════════
# ЧАСТЬ III: ПРИВЯЗКА ГРАФА К Q6 (data7 → pro2)
# ═══════════════════════════════════════════════════════════════════════════

def assign_q6_positions(graph: KnowledgeGraph, model: Variant3GPT,
                        domain_corpus: dict, block_size: int):
    """
    Каждому концепту графа назначаем Q6-позицию.
    Аналог data7: WikiAggregator (нахождение центральных концептов)

    Запускаем model forward на тексте концепта → dominant hexagram.
    """
    model.eval()
    for domain, entries in domain_corpus.items():
        for name, text in entries:
            if name not in graph.concepts:
                continue
            ids = torch.tensor(
                [b for b in text.encode("utf-8")][:block_size],
                dtype=torch.long
            ).unsqueeze(0)
            with torch.no_grad():
                out = model(ids)
                info = out[2] if isinstance(out, tuple) and len(out) > 2 else {}
                hw = info.get("hex_weights")
                if hw is not None:
                    graph.concepts[name].hex_idx = hw.mean(1).argmax(-1)[0].item()


def q6_concept_distances(graph: KnowledgeGraph) -> dict:
    """
    Вычисляем BFS-расстояния между Q6-позициями концептов.
    Аналог data7: cognitive_distance в TSP-оптимизации.
    """
    names = [n for n, c in graph.concepts.items() if c.hex_idx >= 0]
    distances = {}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            ha = graph.concepts[a].hex_idx
            hb = graph.concepts[b].hex_idx
            d  = dist_matrix[ha, hb].item()
            distances[(a, b)] = d
            distances[(b, a)] = d
    return distances


# ═══════════════════════════════════════════════════════════════════════════
# ЧАСТЬ IV: УТИЛИТЫ ОБУЧЕНИЯ
# ═══════════════════════════════════════════════════════════════════════════

def text_to_ids(text, block_size=32):
    ids = [b for b in text.encode("utf-8")][:block_size] or [0]
    return torch.tensor(ids, dtype=torch.long)

def ids_pad(ids_list, block_size):
    out = []
    for ids in ids_list:
        if len(ids) < block_size:
            ids = torch.cat([ids, torch.zeros(block_size - len(ids), dtype=torch.long)])
        out.append(ids[:block_size])
    return torch.stack(out)

def get_hw_dw(model, ids_batch):
    out = model(ids_batch)
    info = out[2] if isinstance(out, tuple) and len(out) > 2 else {}
    hw = info.get("hex_weights")
    if hw is None:
        return None, None
    hw_m = hw.mean(dim=1)
    dw = info.get("domain_weights")
    if dw is not None and dw.dim() == 3:
        dw = dw.mean(1)
    else:
        anchors = [DOMAIN_ANCHORS[d] for d in DOMAINS]
        dw = torch.stack([hw_m[:, a] for a in anchors], dim=-1)
        dw = dw / (dw.sum(-1, keepdim=True) + 1e-8)
    return hw_m, dw

def get_hidden(model, ids):
    emb = model.tok_emb(ids)
    for block in model.blocks:
        emb = block(emb)
    return emb.mean(1)

def make_batch(texts, block_size, batch_size):
    selected = random.sample(texts, min(batch_size, len(texts)))
    xs, ys = [], []
    for t in selected:
        ids = text_to_ids(t, block_size + 1)[:block_size + 1]
        if len(ids) < 2:
            continue
        x, y = ids[:-1], ids[1:]
        if len(x) < block_size:
            pad = torch.zeros(block_size - len(x), dtype=torch.long)
            x, y = torch.cat([x, pad]), torch.cat([y, pad])
        xs.append(x); ys.append(y)
    if not xs:
        return None, None
    return torch.stack(xs), torch.stack(ys)

def make_biangua_batch(batch_size, block_size):
    adj = (biangua > 0.5)
    xs, ys = [], []
    for _ in range(batch_size):
        cur = random.randint(0, 63)
        seq = [cur]
        for _ in range(block_size):
            nbrs = adj[cur].nonzero(as_tuple=False).squeeze(1).tolist()
            cur = random.choice(nbrs) if nbrs else cur
            seq.append(cur)
        xs.append(torch.tensor(seq[:-1], dtype=torch.long))
        ys.append(torch.tensor(seq[1:],  dtype=torch.long))
    return torch.stack(xs), torch.stack(ys)

def get_metrics(model, ids):
    hw, dw = get_hw_dw(model, ids[:1])
    if hw is None:
        return {}
    return evaluator.evaluate(hw.unsqueeze(1), dw)

def gate_entropy_loss(model, ids_batch):
    total = torch.tensor(0.0, requires_grad=True)
    for block in model.blocks:
        gate = block.ternary_gate
        emb = model.tok_emb(ids_batch)
        for b in model.blocks:
            emb = b(emb)
            if b is block:
                break
        scores = torch.tanh(gate.gate_proj(emb) / gate.temperature)
        budget = torch.sigmoid(gate.log_uncertainty)
        thr = (1.0 - budget) * 0.5 + 0.1
        p_y = torch.sigmoid((scores - thr) * 10).mean()
        p_n = torch.sigmoid((-scores - thr) * 10).mean()
        p_0 = (1.0 - p_y - p_n).clamp(0)
        probs = torch.stack([p_y, p_0, p_n]) + 1e-8
        probs = probs / probs.sum()
        entropy = -(probs * probs.log()).sum()
        total = total - entropy
    return total / len(model.blocks)

def domain_triplet_loss(model, domain_corpus, block_size, margin=0.3):
    doms = random.sample(list(domain_corpus.keys()), 2)
    dom_a, dom_b = doms[0], doms[1]
    texts_a = [t for _, t in random.sample(domain_corpus[dom_a],
                                            min(2, len(domain_corpus[dom_a])))]
    texts_b = [t for _, t in random.sample(domain_corpus[dom_b], 1)]
    if len(texts_a) < 2:
        return torch.tensor(0.0)
    batch = ids_pad([text_to_ids(t, block_size) for t in texts_a + texts_b], block_size)
    hw, _ = get_hw_dw(model, batch)
    if hw is None:
        return torch.tensor(0.0)
    a, p, n = hw[0], hw[1], hw[2]
    dist_ap = 1.0 - F.cosine_similarity(a.unsqueeze(0), p.unsqueeze(0))
    dist_an = 1.0 - F.cosine_similarity(a.unsqueeze(0), n.unsqueeze(0))
    return F.relu(dist_ap - dist_an + margin).mean()

def quality_contrastive_loss(model, good_texts, bad_texts, block_size):
    good = random.choice(good_texts)
    bad  = random.choice(bad_texts)
    losses = []
    for txt, target in [(good, 1.0), (bad, 0.0)]:
        ids = text_to_ids(txt, block_size).unsqueeze(0)
        h = get_hidden(model, ids)
        sc, _, _ = qfilter(h.unsqueeze(1))
        prob = ((sc.mean().clamp(-1,1) + 1) / 2.0).unsqueeze(0)
        losses.append(F.binary_cross_entropy(prob, torch.tensor([target])))
    return sum(losses) / len(losses)


# ═══════════════════════════════════════════════════════════════════════════
# ЧАСТЬ V: ДВУНАПРАВЛЕННЫЙ ЦИКЛ ОБУЧЕНИЯ
# ═══════════════════════════════════════════════════════════════════════════

class BidirectionalTrainer:
    """
    Замкнутый цикл: граф ↔ модель.

    ВПЕРЁД:  граф → анкоры → корпус → модель учится
    НАЗАД:   модель генерирует → граф обновляется (AdaptiveLearning)
             → новые пробелы → новые гипотезы → снова в корпус
    """

    def __init__(self, model: Variant3GPT, graph: KnowledgeGraph,
                 domain_corpus: dict, bad_texts: list):
        self.model         = model
        self.graph         = graph
        self.domain_corpus = domain_corpus
        self.bad_texts     = bad_texts
        self.block_size    = model.cfg.block_size

        # Текущий обучающий корпус (пополняется по ходу цикла)
        self.corpus: list[str] = [t for entries in domain_corpus.values()
                                  for _, t in entries]
        # Лог цикла
        self.cycle_log: list[dict] = []

    # ── Вперёд: граф → модель ────────────────────────────────────────────

    def forward_pass(self, steps: int, lr: float,
                     log_every: int = 25) -> dict:
        """
        Обучаем модель на текущем корпусе.
        TSP-порядок концептов используется для формирования батчей.
        """
        # TSP-оптимальный порядок центральных концептов
        tsp_order = self.graph.tsp_order(list(self.graph.concepts.keys()))
        ordered_texts = []
        for name in tsp_order:
            for dom_entries in self.domain_corpus.values():
                for n, t in dom_entries:
                    if n == name:
                        ordered_texts.append(t)
                        break

        # Все тексты (ordered + hypotheses)
        all_texts = ordered_texts + [t for t in self.corpus
                                     if t not in ordered_texts]

        opt = AdamW(list(self.model.parameters()) + list(qfilter.parameters()),
                    lr=lr, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

        losses = []
        for step in range(1, steps + 1):
            self.model.train(); qfilter.train()

            x, y = make_batch(all_texts, self.block_size, 8)
            if x is None:
                continue

            out = self.model(x, y)
            lm = out[1]
            if lm is None:
                logits = out[0]
                lm = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            dom_l  = domain_triplet_loss(self.model, self.domain_corpus, self.block_size)
            qual_l = quality_contrastive_loss(self.model, all_texts, self.bad_texts,
                                              self.block_size)
            gate_l = gate_entropy_loss(self.model, x[:4])
            loss = lm + ALPHA * dom_l + BETA * qual_l + GAMMA * gate_l

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.model.parameters()) +
                                     list(qfilter.parameters()), 1.0)
            opt.step(); sch.step()
            losses.append(loss.item())

            if step % log_every == 0 or step == 1:
                self.model.eval()
                xv, _ = make_batch(all_texts, self.block_size, 4)
                m = get_metrics(self.model, xv) if xv is not None else {}
                hw, dw = get_hw_dw(self.model, (xv if xv is not None
                                                 else x)[:1])
                dh = hw.argmax(-1)[0].item() if hw is not None else 0
                dom = DOMAINS[dw.argmax(-1)[0].item()] if dw is not None else "?"
                print(f"    s{step:4d}  L={loss.item():.3f}  "
                      f"ent={m.get('hex_entropy',0):.3f}  "
                      f"coh={m.get('domain_coherence',0):.3f}  "
                      f"[{dh:2d}]{hname(dh)[:10]}  {dom}")

        return {"steps": steps, "avg_loss": sum(losses)/max(len(losses),1)}

    # ── Назад: модель → граф ─────────────────────────────────────────────

    def backward_pass(self, n_generate: int = 8,
                      temperature: float = 1.1) -> dict:
        """
        Модель генерирует тексты → оцениваем их качество
        → обновляем граф (AdaptiveLearning) → находим новые пробелы
        → генерируем гипотезы → добавляем в корпус.

        Это НЕДОСТАЮЩАЯ ПЕТЛЯ из data7:
          # Missing: proposals → decomposer.decompose() → refinement loop
        """
        self.model.eval(); qfilter.eval()

        generated_texts  = []
        quality_scores   = []
        concept_pairs_activated = []   # пары концептов, активированных в тексте

        for _ in range(n_generate):
            # Промпт = случайный текст из корпуса
            prompt_text = random.choice(self.corpus)
            prompt_ids  = text_to_ids(prompt_text, self.block_size // 2).unsqueeze(0)

            with torch.no_grad():
                gen = prompt_ids.clone()
                for _ in range(self.block_size // 2):
                    out = self.model(gen)
                    logits = out[0] if isinstance(out, tuple) else out
                    probs  = F.softmax(logits[0, -1] / temperature, dim=-1)
                    nxt    = torch.multinomial(probs, 1)
                    gen    = torch.cat([gen, nxt.unsqueeze(0)], dim=1)[:, -self.block_size:]

                # Оцениваем качество
                h = get_hidden(self.model, gen)
                sc, _, _ = qfilter(h.unsqueeze(1))
                score = (sc.clamp(-1,1) + 1).mean().item() / 2.0

                # Q6-позиция сгенерированного текста
                hw, dw = get_hw_dw(self.model, gen)
                gen_hex = hw.argmax(-1)[0].item() if hw is not None else 0

            try:
                gen_text = bytes([b % 256 for b in gen[0].tolist()]).decode(
                    "utf-8", errors="replace")
            except Exception:
                gen_text = ""

            generated_texts.append(gen_text)
            quality_scores.append(score)

            # Найти два концепта, ближайших к Q6-позиции генерации
            best_pair = (None, None)
            min_dist  = 999.0
            for na, ca in self.graph.concepts.items():
                if ca.hex_idx < 0:
                    continue
                for nb, cb in self.graph.concepts.items():
                    if nb == na or cb.hex_idx < 0:
                        continue
                    # Среднее расстояние от пары до gen_hex
                    d = (dist_matrix[ca.hex_idx, gen_hex].item() +
                         dist_matrix[cb.hex_idx, gen_hex].item()) / 2
                    if d < min_dist:
                        min_dist  = d
                        best_pair = (na, nb)
            concept_pairs_activated.append((best_pair, score))

        # AdaptiveLearning: обновляем граф
        updates = 0
        for (pair, score) in concept_pairs_activated:
            if pair[0] and pair[1]:
                # quality > 0.5 → хорошая связь (усиляем), иначе ослабляем
                delta = (score - 0.5) * 2.0   # [-1, +1]
                self.graph.adaptive_update(pair[0], pair[1],
                                           quality_delta=delta, lr=0.05)
                updates += 1

        self.graph.compute_centrality()

        # Найти новые пробелы ПОСЛЕ обновления
        gaps = self.graph.identify_gaps(top_k=8)

        # Генерация гипотез → добавляем в корпус
        hypotheses = self.graph.generate_hypotheses(gaps)
        new_accepted = 0
        for hyp in hypotheses:
            ids = text_to_ids(hyp, self.block_size).unsqueeze(0)
            with torch.no_grad():
                h = get_hidden(self.model, ids)
                sc, _, _ = qfilter(h.unsqueeze(1))
                q = (sc.clamp(-1,1) + 1).mean().item() / 2.0
            if q > 0.50:
                self.corpus.append(hyp)
                new_accepted += 1

        accepted_gen = sum(1 for s in quality_scores if s > 0.52)

        return {
            "generated":      n_generate,
            "accepted_gen":   accepted_gen,
            "graph_updates":  updates,
            "new_gaps":       len(gaps),
            "hypotheses_added": new_accepted,
            "avg_quality":    sum(quality_scores) / len(quality_scores),
            "corpus_size":    len(self.corpus),
        }

    # ── Полный цикл ──────────────────────────────────────────────────────

    def run_cycle(self, n_cycles: int = 4,
                  forward_steps: int = 200,
                  forward_lr: float = 5e-5):
        """
        Главный цикл двунаправленного обучения.

        Каждый цикл:
          1. ВПЕРЁД:  обучаем модель на корпусе (TSP-порядок)
          2. НАЗАД:   модель генерирует → граф обновляется → новые гипотезы
          3. Критерий сходимости: корпус не растёт ИЛИ loss стабилен
        """
        print(f"\n{'═'*72}")
        print(f"  ДВУНАПРАВЛЕННЫЙ ЦИКЛ ОБУЧЕНИЯ")
        print(f"{'═'*72}")
        print(f"  Граф:       {self.graph.summary()['nodes']} концептов, "
              f"{self.graph.summary()['edges']} рёбер")
        print(f"  Корпус:     {len(self.corpus)} текстов")
        print(f"  Циклов:     {n_cycles} × ({forward_steps} шагов вперёд + генерация назад)")

        prev_loss = 9999.0
        for cycle in range(1, n_cycles + 1):
            print(f"\n{'─'*72}")
            print(f"  ЦИКЛ {cycle}/{n_cycles}")
            print(f"{'─'*72}")

            # Обновляем Q6-позиции концептов с актуальными весами модели
            assign_q6_positions(self.graph, self.model, self.domain_corpus,
                                 self.block_size)

            # Показываем ТОП-3 концептов по PageRank
            summ = self.graph.summary()
            print(f"\n  PageRank TOP-3:")
            for name, pr in summ["top_by_pagerank"][:3]:
                c = self.graph.concepts[name]
                print(f"    {name:15s}  PR={pr}  Q6=[{c.hex_idx:2d}]"
                      f" {hname(c.hex_idx) if c.hex_idx>=0 else '?':15s}  "
                      f"домен={c.domain}")

            # Пробелы до обучения
            gaps_before = self.graph.identify_gaps(top_k=3)
            print(f"\n  Пробелы в графе (топ-3):")
            for g in gaps_before:
                print(f"    {g['a']:12s} ↔ {g['b']:12s}  "
                      f"потенциал={g['potential']:.4f}  "
                      f"{'одинаковый домен' if g['same_domain'] else 'разные домены'}")

            # ВПЕРЁД
            print(f"\n  ▶▶ ВПЕРЁД: обучение {forward_steps} шагов "
                  f"(корпус={len(self.corpus)} текстов):")
            fwd = self.forward_pass(forward_steps, forward_lr, log_every=50)

            # НАЗАД
            print(f"\n  ◀◀ НАЗАД: генерация + обновление графа:")
            bwd = self.backward_pass(n_generate=10)
            print(f"    Сгенерировано: {bwd['generated']}  "
                  f"Принято: {bwd['accepted_gen']}  "
                  f"Обновлений графа: {bwd['graph_updates']}")
            print(f"    Новых пробелов: {bwd['new_gaps']}  "
                  f"Гипотез добавлено: {bwd['hypotheses_added']}  "
                  f"Корпус→{bwd['corpus_size']}")
            print(f"    Ср. качество генерации: {bwd['avg_quality']:.4f}")

            # Критерий сходимости
            loss_delta = abs(fwd["avg_loss"] - prev_loss)
            prev_loss  = fwd["avg_loss"]
            converged  = (loss_delta < 0.02 and
                          bwd["hypotheses_added"] == 0)

            self.cycle_log.append({
                "cycle":   cycle,
                "forward": fwd,
                "backward": bwd,
                "converged": converged,
            })

            if converged:
                print(f"\n  ✓ Сходимость на цикле {cycle}: "
                      f"Δloss={loss_delta:.4f}, новых гипотез=0")
                break

        return self.cycle_log


# ═══════════════════════════════════════════════════════════════════════════
# ЧАСТЬ VI: ФИНАЛЬНАЯ ОЦЕНКА
# ═══════════════════════════════════════════════════════════════════════════

def final_evaluation(model, graph, domain_corpus):
    print(f"\n{'═'*72}")
    print(f"  ФИНАЛЬНАЯ ОЦЕНКА ДВУНАПРАВЛЕННОГО ОБУЧЕНИЯ")
    print(f"{'═'*72}")

    model.eval(); qfilter.eval()
    block_size = model.cfg.block_size

    # 1. Доменные расстояния в Q6
    print(f"\n  1. Q6-РАССТОЯНИЯ ВНУТРИ/МЕЖДУ ДОМЕНАМИ (после обучения):")
    domain_pairs = [
        ("fire",  "light",   "PYRO↔PYRO  (один домен)"),
        ("water", "wind",    "HYDRO↔AERO (разные)"),
        ("logic", "pattern", "NOOS↔NOOS  (один домен)"),
        ("star",  "crystal", "COSMO↔GEO  (разные)"),
        ("fire",  "ocean",   "PYRO↔HYDRO (антонимы стихий)"),
    ]
    assign_q6_positions(graph, model, domain_corpus, block_size)
    for a, b, label in domain_pairs:
        if a in graph.concepts and b in graph.concepts:
            ha = graph.concepts[a].hex_idx
            hb = graph.concepts[b].hex_idx
            if ha >= 0 and hb >= 0:
                bfs = dist_matrix[ha, hb].item()
                ids_a = text_to_ids(a, block_size).unsqueeze(0)
                ids_b = text_to_ids(b, block_size).unsqueeze(0)
                hw_a, _ = get_hw_dw(model, ids_a)
                hw_b, _ = get_hw_dw(model, ids_b)
                cos = (F.cosine_similarity(hw_a, hw_b).item()
                       if hw_a is not None else 0)
                print(f"  {label:35s}  [{ha:2d}]↔[{hb:2d}]  "
                      f"BFS={bfs:.0f}  cos={cos:+.4f}")

    # 2. Качество генерации до/после
    print(f"\n  2. РАЗРЫВ КАЧЕСТВА (хорошее − плохое):")
    test_pairs = [
        ("Fire transforms matter through energy release.", "spam buy now click here"),
        ("Logic and reason guide systematic thought.",     "aaaaaaaaaaaaaaaaaaa"),
        ("Stars synthesize elements through fusion.",      "x x x x x x x x x"),
    ]
    for good, bad in test_pairs:
        ids_g = text_to_ids(good, block_size).unsqueeze(0)
        ids_b = text_to_ids(bad,  block_size).unsqueeze(0)
        with torch.no_grad():
            hg = get_hidden(model, ids_g)
            hb = get_hidden(model, ids_b)
            sg, _, _ = qfilter(hg.unsqueeze(1))
            sb, _, _ = qfilter(hb.unsqueeze(1))
            diff = sg.mean().item() - sb.mean().item()
        mark = "✓" if diff > 0 else "✗"
        print(f"  {mark} diff={diff:+.4f}  '{good[:35]}' vs '{bad[:25]}'")

    # 3. PageRank TOP после всего обучения
    print(f"\n  3. ГРАФ ПОСЛЕ ОБУЧЕНИЯ (PageRank TOP-5):")
    graph.compute_centrality()
    summ = graph.summary()
    for name, pr in summ["top_by_pagerank"]:
        c = graph.concepts[name]
        print(f"  {name:15s}  PR={pr}  Q6=[{c.hex_idx:2d}]"
              f" {hname(c.hex_idx) if c.hex_idx>=0 else '?':15s}  "
              f"домен={c.domain}  depth={c.depth:.2f}")

    # 4. Пробелы ПОСЛЕ обучения — что ещё не выучено
    print(f"\n  4. ОСТАТОЧНЫЕ ПРОБЕЛЫ (граф не объяснил модели):")
    gaps_after = graph.identify_gaps(top_k=5)
    for g in gaps_after:
        print(f"  {g['a']:12s} ↔ {g['b']:12s}  потенциал={g['potential']:.4f}  "
              f"→ следующий цикл обучения")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  BIDIR TRAIN — ДВУНАПРАВЛЕННЫЙ ЦИКЛ ОБУЧЕНИЯ")
    print("  data7 (граф знаний) ⇄ pro2 (Variant3GPT)")
    print("=" * 72)
    print(f"\n  Реализует НЕДОСТАЮЩУЮ ПЕТЛЮ из data7/knowledge_transformer.py:")
    print(f"  # Missing: proposals → decomposer.decompose() → refinement loop")
    print(f"\n  Теперь:")
    print(f"  граф→гипотезы → модель учится → модель генерирует →")
    print(f"  граф обновляется (AdaptiveLearning) → новые пробелы →")
    print(f"  новые гипотезы → снова в обучение (цикл замкнут)")

    # Инициализация
    model = Variant3GPT(CFG).to(DEVICE)
    print(f"\n  Модель: {model.count_parameters():,} параметров")

    # Строим граф знаний
    print(f"\n  Строим KnowledgeGraph из {len(DOMAIN_CORPUS)} доменов...")
    graph = build_initial_graph(DOMAIN_CORPUS)
    summ  = graph.summary()
    print(f"  Граф: {summ['nodes']} концептов, {summ['edges']} рёбер")
    print(f"  PageRank TOP-3: {summ['top_by_pagerank'][:3]}")

    # Стадия 0: самопознание (biangua без текстов)
    print(f"\n  Стадия 0: самопознание Q6 (200 шагов)...")
    opt0 = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for step in range(1, 201):
        model.train()
        x, y = make_biangua_batch(16, CFG.block_size)
        out = model(x, y)
        lm = out[1]
        if lm is None:
            logits = out[0]
            lm = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        gl = gate_entropy_loss(model, x[:4])
        loss = lm + GAMMA * gl
        opt0.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt0.step()
        if step % 50 == 0:
            model.eval()
            xv, _ = make_biangua_batch(4, CFG.block_size)
            with torch.no_grad():
                out = model(xv)
                info = out[2] if isinstance(out, tuple) else {}
                hw = info.get("hex_weights")
                yang = yin = 0.0
                if hw is not None:
                    for block in model.blocks:
                        g = block.ternary_gate
                        s = torch.tanh(g.gate_proj(model.tok_emb(xv)) / g.temperature)
                        budget = torch.sigmoid(g.log_uncertainty)
                        thr = (1 - budget) * 0.5 + 0.1
                        yang += (s > thr).float().mean().item()
                        yin  += (s < -thr).float().mean().item()
                    yang /= len(model.blocks)
                    yin  /= len(model.blocks)
            print(f"  s{step:4d}  loss={loss.item():.3f}  "
                  f"▲{yang*100:.0f}%◼{(1-yang-yin)*100:.0f}%▽{yin*100:.0f}%")

    # Назначаем Q6-позиции концептам
    assign_q6_positions(graph, model, DOMAIN_CORPUS, CFG.block_size)

    # Запускаем двунаправленный тренер
    trainer = BidirectionalTrainer(model, graph, DOMAIN_CORPUS, BAD_TEXTS)
    cycle_log = trainer.run_cycle(
        n_cycles=4,
        forward_steps=150,
        forward_lr=3e-5,
    )

    # Финальная оценка
    final_evaluation(model, graph, DOMAIN_CORPUS)

    # Сохранение
    torch.save({
        "model_state":   model.state_dict(),
        "qfilter_state": qfilter.state_dict(),
        "config":        CFG.__dict__,
        "corpus_size":   len(trainer.corpus),
    }, "checkpoint_bidir.pt")

    with open("bidir_train_log.json", "w") as f:
        json.dump({
            "cycle_log":    cycle_log,
            "final_corpus": len(trainer.corpus),
            "graph_nodes":  graph.summary()["nodes"],
            "graph_edges":  graph.summary()["edges"],
        }, f, indent=2, default=str)

    print(f"\n  Чекпоинт : checkpoint_bidir.pt")
    print(f"  Лог      : bidir_train_log.json")
    print(f"  Корпус   : {len(trainer.corpus)} текстов (начало: 30)")
    print(f"\n  Цикл замкнут. Недостающая петля data7 реализована.")


if __name__ == "__main__":
    main()

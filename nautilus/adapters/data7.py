"""
Data7Adapter — адаптер для svend4/data7.

data7 = теоретическая база:
  - knowledge_transformation_theory.md: Диссертации ⇄ Энциклопедии
  - knowledge_transformer.py: K₀→K₁→K₂ цикл
  - НЕДОСТАЮЩАЯ ПЕТЛЯ реализована в pro2/bidir_train.py

Статические записи: data7 не имеет публичного API,
поэтому адаптер описывает его теоретические концепты.
"""

from .base import BaseAdapter, PortalEntry


class Data7Adapter(BaseAdapter):
    name = "data7"
    REPO = "svend4/data7"

    def fetch(self, query: str) -> list[PortalEntry]:
        q = query.lower()
        entries = self._all_entries()
        if q in ("knowledge", "all", ""):
            return entries
        return [e for e in entries if q in e.title.lower() or q in e.content.lower()]

    def _all_entries(self) -> list[PortalEntry]:
        return [
            PortalEntry(
                id="data7:theory:transformation",
                title="Теория трансформации знаний K₀→K₁→K₂",
                source=self.REPO,
                format_type="theory",
                content=(
                    "Диссертации ⇄ Энциклопедии. "
                    "K₀ → decompose → aggregate → K₁ → decompose → synthesize → K₂ → ... "
                    "Итеративное сжатие и расширение знаний между специализированным "
                    "и обобщённым полюсами."
                ),
                metadata={
                    "file": "knowledge_transformation_theory.md",
                    "cycle": "K₀→K₁→K₂",
                    "poles": ["dissertations", "encyclopedias"],
                },
                links=["pro2:bidir", "pro2:knowledge_graph"],
            ),
            PortalEntry(
                id="data7:missing_loop",
                title="Недостающая петля (реализована в pro2)",
                source=self.REPO,
                format_type="theory",
                content=(
                    "# Missing: proposals → decomposer.decompose() → refinement loop\n"
                    "Комментарий в data7/knowledge_transformer.py.\n"
                    "Эта петля реализована в pro2/bidir_train.py:\n"
                    "GPT генерирует гипотезы → граф оценивает → AdaptiveLearning "
                    "обновляет веса → identify_gaps() → новый корпус → снова вперёд."
                ),
                metadata={
                    "file": "knowledge_transformer.py",
                    "status": "missing_in_data7",
                    "implemented_in": "pro2/bidir_train.py",
                },
                links=["pro2:bidir", "pro2:adaptive_learning"],
            ),
            PortalEntry(
                id="data7:concept",
                title="Concept — атомарная единица знания",
                source=self.REPO,
                format_type="schema",
                content=(
                    "Базовый класс data7. В pro2 расширен: "
                    "Concept.hex_idx (Q6-позиция), "
                    "Concept.pagerank (PageRank-центральность в графе), "
                    "Concept.depth (специализация 0..1), "
                    "Concept.novelty (новизна 0..1)."
                ),
                metadata={"class": "Concept", "extended_in": "pro2/bidir_train.py"},
                links=["pro2:knowledge_graph", "pro2:q6"],
            ),
            PortalEntry(
                id="data7:analogy_table",
                title="Таблица аналогий data7 ↔ pro2",
                source=self.REPO,
                format_type="bridge",
                content=(
                    "KnowledgeGraph.compute_centrality()  ↔  hex_weights (мягкое внимание)\n"
                    "identify_gaps()                       ↔  domain_triplet_loss (разрыв)\n"
                    "generate_hypotheses()                 ↔  self_dialog stage 3\n"
                    "AdaptiveLearningOptimizer             ↔  gradient descent на QFilter\n"
                    "TSP-оптимизация порядка               ↔  BFS-путь по biangua-графу"
                ),
                metadata={"type": "analogy_table", "source_file": "bidir_train.py docstring"},
                links=["pro2:bidir", "pro2:knowledge_graph"],
            ),
        ]

    def describe(self) -> dict:
        return {
            "repo": self.REPO,
            "format": "data7",
            "native_unit": "Концепт + граф знаний + цикл трансформации",
            "abstraction_range": "K₀ (специализированное) ↔ K∞ (обобщённое)",
            "key_insight": "Диссертации ⇄ Энциклопедии как полюса знания",
            "missing_loop_implemented_in": "pro2/bidir_train.py",
        }

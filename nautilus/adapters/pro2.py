"""
Pro2Adapter — адаптер для svend4/pro2.
Формат: Q6-концепты (6-битное семантическое пространство), граф знаний.
"""

import json
import re
from pathlib import Path
from .base import BaseAdapter, PortalEntry


_REPO_ROOT = Path(__file__).parent.parent.parent


class Pro2Adapter(BaseAdapter):
    name = "pro2"
    REPO = "svend4/pro2"

    _LOCAL_LOG = _REPO_ROOT / "bidir_train_v2_log.json"
    _SELF_IMPROVEMENT = (
        _REPO_ROOT / "data" / "svend4_corpus" / "infosystems" / "info1" /
        "self_improvement_report.txt"
    )

    def fetch(self, query: str) -> list[PortalEntry]:
        results = []
        if self._LOCAL_LOG.exists():
            try:
                data = json.loads(self._LOCAL_LOG.read_text())
                results = self._search_log(query, data)
            except Exception:
                pass
        return results or self._static_entries()

    def _search_log(self, query: str, log: dict) -> list[PortalEntry]:
        results = []
        for key in ["final_stats", "training_summary", "domain_coverage"]:
            if key in log:
                results.append(PortalEntry(
                    id=f"pro2:log:{key}",
                    title=f"pro2 / {key}",
                    source=self.REPO,
                    format_type="concept",
                    content=str(log[key])[:300],
                    metadata={"log_key": key},
                    links=["info1:methodology", "meta:hexagram:50"],
                ))
        return results[:3]

    def _parse_self_improvement(self) -> dict | None:
        if not self._SELF_IMPROVEMENT.exists():
            return None
        text = self._SELF_IMPROVEMENT.read_text(encoding="utf-8")
        metrics = {}
        for key in ["CD", "VT", "CR", "DB"]:
            m = re.search(rf"{key}:\s+([\d.]+)%?", text)
            if m:
                metrics[key] = float(m.group(1))
        m_cycles = re.search(r"Циклов:\s+(\d+)", text)
        m_docs = re.search(r"Документов:\s+(\d+)", text)
        m_links = re.search(r"Связей:\s+(\d+)", text)
        if m_cycles:
            metrics["cycles"] = int(m_cycles.group(1))
        if m_docs:
            metrics["documents"] = int(m_docs.group(1))
        if m_links:
            metrics["connections"] = int(m_links.group(1))
        return metrics if metrics else None

    def _static_entries(self) -> list[PortalEntry]:
        entries = [
            PortalEntry(
                id="pro2:q6",
                title="Q6 Семантическое пространство",
                source=self.REPO,
                format_type="concept",
                content=(
                    "64 состояния (6-битное пространство). "
                    "Каждый концепт = координата Q6[b0..b5]. "
                    "Соответствует 64 гексаграммам И-Цзин."
                ),
                metadata={"dims": 6, "states": 64},
                links=["meta:hexagram:all", "info1:alpha:0"],
            ),
            PortalEntry(
                id="pro2:bidir",
                title="Bidirectional Training ⇑⇓ (замкнутый цикл)",
                source=self.REPO,
                format_type="concept",
                content=(
                    "ВПЕРЁД (специализация→обобщение): "
                    "Корпус→KnowledgeGraph→PageRank-центры→Q6-анкоры→Variant3GPT. "
                    "НАЗАД (обобщение→специализация): "
                    "GPT генерирует→QFilter оценивает→AdaptiveLearning обновляет граф"
                    "→identify_gaps()→generate_hypotheses()→новый корпус→снова вперёд. "
                    "Реализует НЕДОСТАЮЩУЮ ПЕТЛЮ из data7/knowledge_transformer.py."
                ),
                metadata={
                    "method": "bidirectional",
                    "file": "bidir_train.py",
                    "criterion": "модель генерирует тексты, граф признаёт 'достаточно центральными'",
                    "analogy": {
                        "data7:compute_centrality": "hex_weights",
                        "data7:identify_gaps": "domain_triplet_loss",
                        "data7:generate_hypotheses": "self_dialog stage 3",
                    },
                },
                links=["info1:methodology", "data7:missing_loop", "data7:theory:transformation"],
            ),
            PortalEntry(
                id="pro2:knowledge_graph",
                title="KnowledgeGraph — граф научных знаний",
                source=self.REPO,
                format_type="concept",
                content=(
                    "Аналог data7:KnowledgeGraph. "
                    "Отличие: рёбра взвешены качеством связи (обновляется AdaptiveLearning). "
                    "Типы рёбер: causes, extends, contradicts, related_to, is_a, part_of. "
                    "PageRank-центральность определяет Q6-анкоры для обучения."
                ),
                metadata={"file": "bidir_train.py", "class": "KnowledgeGraph"},
                links=["data7:concept", "data7:theory:transformation"],
            ),
            PortalEntry(
                id="pro2:adaptive_learning",
                title="AdaptiveLearning — динамический выбор концептов",
                source=self.REPO,
                format_type="concept",
                content=(
                    "Обновляет веса рёбер графа по результатам генерации GPT. "
                    "identify_gaps() ↔ domain_triplet_loss (из data7). "
                    "generate_hypotheses() ↔ self_dialog stage 3. "
                    "Критерий завершения: L_total = L_lm + α·L_domain + β·L_quality + γ·L_gate."
                ),
                metadata={
                    "file": "bidir_train.py",
                    "loss": "L_lm + 0.30*L_domain + 0.20*L_quality + 0.10*L_gate",
                },
                links=["data7:missing_loop", "pro2:bidir"],
            ),
        ]

        # Добавляем запись с метриками самосовершенствования если доступны
        si = self._parse_self_improvement()
        if si:
            cd_status = "⚠️" if si.get("CD", 0) > 20 else "✅"
            vt_status = "⚠️" if si.get("VT", 100) < 50 else "✅"
            entries.append(PortalEntry(
                id="pro2:self_improvement",
                title="Self-Improvement Metrics (info1, этап 6)",
                source=self.REPO,
                format_type="metrics",
                content=(
                    f"Документов: {si.get('documents', '?')} · "
                    f"Связей: {si.get('connections', '?')} · "
                    f"Циклов: {si.get('cycles', '?')}\n"
                    f"CD: {si.get('CD', '?')}% {cd_status} (цель: 20%) · "
                    f"VT: {si.get('VT', '?')}% {vt_status} (цель: 50%) · "
                    f"CR: {si.get('CR', '?')} ✅ · DB: {si.get('DB', '?')}% ✅\n"
                    f"Рекомендации: усилить ⇑⇓ связи, сократить ↔, "
                    f"создать ВЕРТИКАЛЬНАЯ-ТРАССИРУЕМОСТЬ.md"
                ),
                metadata=si,
                links=["info1:methodology", "pro2:bidir"],
            ))

        return entries

    def describe(self) -> dict:
        return {
            "repo": self.REPO,
            "format": "pro2",
            "native_unit": "Концепт с Q6-координатой",
            "abstraction_range": "Q6[b0..b5] = 64 состояния",
            "semantic_space": "6D бинарное пространство",
            "hexagram_mapping": "64 гексаграммы И-Цзин",
            "bidir_cycle": "bidir_train.py — замкнутый цикл (реализует data7 missing loop)",
        }

"""
Pro2Adapter — адаптер для svend4/pro2.
Формат: Q6-концепты (6-битное семантическое пространство), граф знаний.
"""

import json
from pathlib import Path
from .base import BaseAdapter, PortalEntry


class Pro2Adapter(BaseAdapter):
    name = "pro2"
    REPO = "svend4/pro2"

    # Путь к локальным данным (относительно этого файла)
    _LOCAL_LOG = Path(__file__).parent.parent.parent / "bidir_train_v2_log.json"

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

    def _static_entries(self) -> list[PortalEntry]:
        return [
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
                title="Bidirectional Training ⇑⇓",
                source=self.REPO,
                format_type="concept",
                content=(
                    "Обучение ⇓ генерация текста из концептов, "
                    "⇑ извлечение концептов из текста. "
                    "Аналог ⇑⇓ методологии info1."
                ),
                metadata={"method": "bidirectional"},
                links=["info1:methodology"],
            ),
        ]

    def describe(self) -> dict:
        return {
            "repo": self.REPO,
            "format": "pro2",
            "native_unit": "Концепт с Q6-координатой",
            "abstraction_range": "Q6[b0..b5] = 64 состояния",
            "semantic_space": "6D бинарное пространство",
            "hexagram_mapping": "64 гексаграммы И-Цзин",
        }

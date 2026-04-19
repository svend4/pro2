"""
Info1Adapter — адаптер для svend4/info1.
Формат: Markdown-документы с α-уровнями абстракции (-4..+4).
"""

import json
import os
import urllib.parse
import urllib.request
from .base import BaseAdapter, PortalEntry


class Info1Adapter(BaseAdapter):
    name = "info1"
    REPO = "svend4/info1"

    ALPHA_MAP = {
        "онтология": +4, "философия": +3, "методология": +2,
        "руководства": +1, "концепция": 0, "спецификации": -1,
        "реализация": -2, "примеры": -3, "код": -4,
    }

    def _github_headers(self) -> dict:
        headers = {"User-Agent": "nautilus-portal/1.0"}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def fetch(self, query: str) -> list[PortalEntry]:
        try:
            url = (
                f"https://api.github.com/search/code"
                f"?q={urllib.parse.quote(query)}+repo:{self.REPO}+language:Markdown"
            )
            req = urllib.request.Request(url, headers=self._github_headers())
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            results = []
            for item in data.get("items", [])[:5]:
                alpha = self._guess_alpha(item["path"])
                results.append(PortalEntry(
                    id=f"info1:{item['sha'][:8]}",
                    title=item["name"],
                    source=self.REPO,
                    format_type="document",
                    content=item.get("path", ""),
                    metadata={"alpha": alpha, "path": item["path"]},
                    links=[f"pro2:depth:{abs(alpha)}"],
                ))
            return results or self._static_entries()
        except Exception:
            return self._static_entries()

    def _guess_alpha(self, path: str) -> int:
        for keyword, alpha in self.ALPHA_MAP.items():
            if keyword in path.lower():
                return alpha
        return 0

    def _static_entries(self) -> list[PortalEntry]:
        return [
            PortalEntry(
                id="info1:methodology",
                title="Методология ⇑⇓↔",
                source=self.REPO,
                format_type="document",
                content=(
                    "Параллельное двунаправленное развитие. "
                    "8 уровней абстракции (α=-4..+4). 74 документа, 1156 связей."
                ),
                metadata={"alpha": +2, "path": "README.md#methodology"},
                links=["pro2:bidir_train", "meta:hexagram:50"],
            ),
            PortalEntry(
                id="info1:cards",
                title="Карточная система",
                source=self.REPO,
                format_type="document",
                content=(
                    "Карточки как атомарные единицы знания. "
                    "8 типов карточек. Аналог концептов Q6 в pro2."
                ),
                metadata={"alpha": -1, "path": "02-Информационная-система/"},
                links=["pro2:concept:knowledge"],
            ),
        ]

    def describe(self) -> dict:
        return {
            "repo": self.REPO,
            "format": "info1",
            "native_unit": "Markdown-документ с α-уровнем",
            "abstraction_range": "α от -4 (код) до +4 (онтология)",
            "total_docs": "74+",
            "total_links": 1156,
        }

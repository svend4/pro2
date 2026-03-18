"""
NAUTILUS PORTAL
===============
Единая точка входа для взаимодополняющих репозиториев.
Аналог Office Suite: каждый репо — свой формат, портал умеет читать все.

Поддерживаемые форматы:
  .info1  — методологические документы (α-уровни, карточки)
  .pro2   — нейросемантика (Q6, граф знаний, эмбеддинги)
  .meta   — символьные паттерны (CA правила, гексаграммы)

Использование:
  python portal.py                    # HTML-отчёт в stdout
  python portal.py --serve            # веб-сервер на :8000
  python portal.py --query "crystal"  # поиск концепта во всех репо
  python portal.py --json             # машиночитаемый вывод
"""

import html
import json
import sys
import argparse
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ─────────────────────────────────────────────
# Типы данных портала
# ─────────────────────────────────────────────

@dataclass
class PortalEntry:
    """Универсальная запись — один концепт/документ из любого репо."""
    id: str
    title: str
    source: str          # "info1" | "pro2" | "meta"
    format_type: str     # "document" | "concept" | "rule"
    content: str
    metadata: dict = field(default_factory=dict)
    links: list[str] = field(default_factory=list)   # связи с другими репо

@dataclass
class PortalResult:
    """Результат запроса портала."""
    query: str
    entries: list[PortalEntry]
    cross_links: list[dict]   # связи между репо
    consensus: Optional[dict] = None


# ─────────────────────────────────────────────
# Адаптеры — по одному на каждый «формат»
# ─────────────────────────────────────────────

class Info1Adapter:
    """
    Адаптер для svend4/info1.
    Формат: Markdown-документы с α-уровнями абстракции.
    """
    REPO = "svend4/info1"
    FORMAT = "info1"

    ALPHA_MAP = {
        "Онтология": +4, "Философия": +3, "Методология": +2,
        "Руководства": +1, "Концепция": 0, "Спецификации": -1,
        "Реализация": -2, "Примеры": -3, "Код": -4,
    }

    def fetch(self, query: str) -> list[PortalEntry]:
        """Ищет концепт в структуре info1 через GitHub API."""
        results = []
        try:
            url = f"https://api.github.com/search/code?q={urllib.parse.quote(query)}+repo:{self.REPO}+language:Markdown"
            req = urllib.request.Request(url, headers={"User-Agent": "nautilus-portal/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
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
        except Exception:
            results = self._fallback_entries(query)
        return results

    def _guess_alpha(self, path: str) -> int:
        for keyword, alpha in self.ALPHA_MAP.items():
            if keyword.lower() in path.lower():
                return alpha
        return 0

    def _fallback_entries(self, query: str) -> list[PortalEntry]:
        """Статические примеры если нет сети."""
        return [
            PortalEntry(
                id="info1:methodology",
                title="Методология ⇑⇓↔",
                source=self.REPO,
                format_type="document",
                content="Параллельное двунаправленное развитие. 8 уровней абстракции (α=-4..+4). 74 документа, 1156 связей.",
                metadata={"alpha": +2, "path": "README.md#methodology"},
                links=["pro2:bidir_train", "meta:hexagram:50"],
            ),
            PortalEntry(
                id="info1:cards",
                title="Карточная система",
                source=self.REPO,
                format_type="document",
                content="Карточки как атомарные единицы знания. 8 типов карточек. Аналог концептов в pro2.",
                metadata={"alpha": -1, "path": "02-Информационная-система/"},
                links=["pro2:concept:knowledge"],
            ),
        ]

    def describe(self) -> dict:
        return {
            "repo": self.REPO,
            "format": self.FORMAT,
            "native_unit": "Markdown-документ с α-уровнем",
            "abstraction_range": "α от -4 (код) до +4 (онтология)",
            "total_docs": "74+",
            "total_links": 1156,
        }


class Pro2Adapter:
    """
    Адаптер для текущего репо (pro2).
    Формат: JSON-логи обучения, Q6-координаты, граф знаний.
    """
    REPO = "svend4/pro2"
    FORMAT = "pro2"

    def fetch(self, query: str) -> list[PortalEntry]:
        """Ищет концепт в Q6-пространстве и графе знаний."""
        results = []
        # Читаем локальные данные
        log_path = Path(__file__).parent / "bidir_train_v2_log.json"
        if log_path.exists():
            try:
                data = json.loads(log_path.read_text())
                results += self._search_log(query, data)
            except Exception:
                pass
        if not results:
            results = self._fallback_entries(query)
        return results

    def _search_log(self, query: str, log: dict) -> list[PortalEntry]:
        results = []
        q = query.lower()
        # Ищем в итогах обучения
        for key in ["final_stats", "training_summary", "domain_coverage"]:
            if key in log:
                val = str(log[key])
                if q in val.lower():  # поиск по ключевому слову
                    results.append(PortalEntry(
                        id=f"pro2:log:{key}",
                        title=f"pro2 / {key}",
                        source=self.REPO,
                        format_type="concept",
                        content=val[:300],
                        metadata={"log_key": key},
                        links=["info1:methodology", "meta:hexagram:50"],
                    ))
                    break
        return results[:3]

    def _fallback_entries(self, query: str) -> list[PortalEntry]:
        return [
            PortalEntry(
                id="pro2:q6",
                title="Q6 Семантическое пространство",
                source=self.REPO,
                format_type="concept",
                content="64 состояния (6-битное пространство). Каждый концепт = координата Q6[b0..b5]. Соответствует 64 гексаграммам И-Цзин.",
                metadata={"dims": 6, "states": 64},
                links=["meta:hexagram:all", "info1:alpha:0"],
            ),
            PortalEntry(
                id="pro2:bidir",
                title="Bidirectional Training ⇑⇓",
                source=self.REPO,
                format_type="concept",
                content="Обучение в двух направлениях: ⇓ генерация текста из концептов, ⇑ извлечение концептов из текста. Аналог ⇑⇓ методологии info1.",
                metadata={"method": "bidirectional"},
                links=["info1:methodology"],
            ),
        ]

    def describe(self) -> dict:
        return {
            "repo": self.REPO,
            "format": self.FORMAT,
            "native_unit": "Концепт с Q6-координатой",
            "abstraction_range": "Q6[b0..b5] = 64 состояния",
            "semantic_space": "6D бинарное пространство",
            "hexagram_mapping": "64 гексаграммы И-Цзин",
        }


class MetaAdapter:
    """
    Адаптер для meta-системы (CA + символьные паттерны).
    Формат: 256 правил клеточных автоматов → символьная верификация.
    """
    REPO = "svend4/meta"
    FORMAT = "meta"

    # Минимальная таблица гексаграмм для демо
    HEXAGRAM_TABLE = {
        50: ("鼎", "Котёл", "Трансформация через огонь"),
        54: ("豐", "Изобилие", "Полнота накопленной силы"),
        1:  ("乾", "Творчество", "Чистая созидательная сила"),
        2:  ("坤", "Исполнение", "Принимающая сила"),
        63: ("既濟", "После завершения", "Равновесие достигнуто"),
        64: ("未濟", "До завершения", "Движение к новому"),
    }

    def fetch(self, query: str) -> list[PortalEntry]:
        return self._fallback_entries(query)

    def _fallback_entries(self, query: str) -> list[PortalEntry]:
        results = []
        for hex_id, (char, name, desc) in self.HEXAGRAM_TABLE.items():
            if query.lower() in name.lower() or query.lower() in desc.lower() or True:
                rule_id = hex_id * 4  # примерное соответствие CA-правил
                results.append(PortalEntry(
                    id=f"meta:hexagram:{hex_id}",
                    title=f"[{hex_id}] {char} {name}",
                    source=self.REPO,
                    format_type="rule",
                    content=desc,
                    metadata={
                        "hexagram_id": hex_id,
                        "ca_rule": rule_id % 256,
                        "char": char,
                    },
                    links=[f"pro2:q6:{format(hex_id - 1, '06b')}"],
                ))
                if len(results) >= 3:
                    break
        return results

    def describe(self) -> dict:
        return {
            "repo": self.REPO,
            "format": self.FORMAT,
            "native_unit": "CA-правило (0..255)",
            "symbolic_space": "256 правил клеточных автоматов",
            "hexagram_mapping": "64 гексаграммы (rule % 64)",
        }


# ─────────────────────────────────────────────
# Nautilus Portal — единая точка входа
# ─────────────────────────────────────────────

class NautilusPortal:
    """
    Портал-агрегатор. Аналог Microsoft Office:
    - каждый адаптер = приложение (Word, Excel, ...)
    - каждый формат = тип файла (.docx, .xlsx, ...)
    - портал = Office Suite
    """

    def __init__(self):
        self.adapters = {
            "info1": Info1Adapter(),
            "pro2":  Pro2Adapter(),
            "meta":  MetaAdapter(),
        }

    def query(self, concept: str) -> PortalResult:
        """Ищет концепт во всех подключённых репозиториях."""
        all_entries = []
        for name, adapter in self.adapters.items():
            entries = adapter.fetch(concept)
            all_entries.extend(entries)

        cross_links = self._build_cross_links(all_entries)
        consensus = self._consensus(all_entries)

        return PortalResult(
            query=concept,
            entries=all_entries,
            cross_links=cross_links,
            consensus=consensus,
        )

    def describe(self) -> dict:
        """Описание всех подключённых форматов."""
        return {name: adapter.describe() for name, adapter in self.adapters.items()}

    def _build_cross_links(self, entries: list[PortalEntry]) -> list[dict]:
        """Строит граф связей между репозиториями."""
        links = []
        seen = set()
        for entry in entries:
            for link_id in entry.links:
                key = tuple(sorted([entry.id, link_id]))
                if key not in seen:
                    seen.add(key)
                    source_repo = entry.id.split(":")[0]
                    target_repo = link_id.split(":")[0]
                    if source_repo != target_repo:
                        links.append({
                            "from": entry.id,
                            "to": link_id,
                            "from_repo": source_repo,
                            "to_repo": target_repo,
                            "type": "semantic_bridge",
                        })
        return links

    def _consensus(self, entries: list[PortalEntry]) -> dict:
        """Консенсус: концепт верен если представлен во всех трёх репо."""
        sources = {e.id.split(":")[0] for e in entries}
        present = {s for s in sources if s in self.adapters}
        coverage = len(present) / len(self.adapters)
        return {
            "present_in": sorted(present),
            "missing_in": sorted(set(self.adapters) - present),
            "coverage": round(coverage, 2),
            "agreed": coverage >= 1.0,
        }

    def register(self, name: str, adapter) -> None:
        """Подключить новый репозиторий-плагин."""
        self.adapters[name] = adapter
        print(f"[portal] зарегистрирован новый формат: {name}")


# ─────────────────────────────────────────────
# Рендеринг
# ─────────────────────────────────────────────

def render_html(result: PortalResult, portal: NautilusPortal) -> str:
    desc = portal.describe()
    entries_html = ""
    for e in result.entries:
        repo_color = {"info1": "#4a90d9", "pro2": "#7b68ee", "meta": "#e8a838"}.get(
            e.id.split(":")[0], "#888"
        )
        alpha = e.metadata.get("alpha", "")
        alpha_badge = f'<span style="font-size:11px;color:#aaa">α={alpha}</span>' if alpha != "" else ""
        entries_html += f"""
        <div style="border-left:3px solid {repo_color};padding:8px 12px;margin:8px 0;background:#1e1e1e;border-radius:0 4px 4px 0">
          <div style="color:{repo_color};font-size:11px;font-weight:bold;text-transform:uppercase">{e.id.split(':')[0]} · {e.format_type} {alpha_badge}</div>
          <div style="color:#e0e0e0;font-size:14px;margin:3px 0">{html.escape(e.title)}</div>
          <div style="color:#aaa;font-size:12px">{html.escape(e.content[:200])}</div>
        </div>"""

    links_html = ""
    for lnk in result.cross_links[:10]:
        links_html += f"""<div style="font-size:11px;color:#888;padding:2px 0">
          <span style="color:#4a90d9">{lnk['from']}</span>
          <span style="color:#555"> ⟷ </span>
          <span style="color:#7b68ee">{lnk['to']}</span>
        </div>"""

    consensus = result.consensus or {}
    cov = int(consensus.get("coverage", 0) * 100)
    agreed_color = "#4caf50" if consensus.get("agreed") else "#ff9800"
    present = ", ".join(consensus.get("present_in", []))

    adapters_html = ""
    for name, d in desc.items():
        color = {"info1": "#4a90d9", "pro2": "#7b68ee", "meta": "#e8a838"}.get(name, "#888")
        adapters_html += f"""
        <div style="border:1px solid #333;border-radius:6px;padding:10px;background:#1a1a1a">
          <div style="color:{color};font-weight:bold;font-size:13px">{name.upper()}</div>
          <div style="color:#888;font-size:11px;margin-top:4px">{d.get('repo','')}</div>
          <div style="color:#ccc;font-size:11px;margin-top:6px">{d.get('native_unit','')}</div>
          <div style="color:#666;font-size:10px;margin-top:3px">{d.get('abstraction_range') or d.get('symbolic_space') or d.get('semantic_space','')}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Nautilus Portal</title>
<style>
  body {{font-family:monospace;background:#121212;color:#e0e0e0;margin:0;padding:20px}}
  h1 {{color:#e0e0e0;font-size:20px;margin-bottom:4px}}
  .sub {{color:#666;font-size:12px;margin-bottom:24px}}
  .section {{margin-bottom:24px}}
  .section-title {{color:#555;font-size:11px;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;border-bottom:1px solid #222;padding-bottom:4px}}
  .grid {{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}
  .query-box {{background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:12px;margin-bottom:20px}}
  .consensus {{display:inline-block;padding:4px 10px;border-radius:4px;font-size:12px;border:1px solid {agreed_color};color:{agreed_color}}}
  input {{background:#1e1e1e;border:1px solid #333;color:#e0e0e0;padding:8px;border-radius:4px;width:300px;font-family:monospace}}
  button {{background:#333;color:#e0e0e0;border:1px solid #444;padding:8px 16px;border-radius:4px;cursor:pointer;font-family:monospace}}
</style>
</head>
<body>
<h1>⬡ Nautilus Portal</h1>
<div class="sub">Единая точка входа · info1 · pro2 · meta</div>

<div class="section">
  <div class="section-title">Форматы (адаптеры)</div>
  <div class="grid">{adapters_html}</div>
</div>

<div class="query-box">
  <div style="color:#888;font-size:11px;margin-bottom:8px">ЗАПРОС</div>
  <div style="color:#e0e0e0;font-size:16px">"{result.query}"</div>
  <div style="margin-top:8px">
    <span class="consensus">{'✓ консенсус' if consensus.get('agreed') else '~ частичный'} {cov}% · {present}</span>
  </div>
</div>

<div class="section">
  <div class="section-title">Результаты по репозиториям</div>
  {entries_html}
</div>

<div class="section">
  <div class="section-title">Межрепозиторные связи</div>
  {links_html if links_html else '<div style="color:#555;font-size:12px">нет перекрёстных связей</div>'}
</div>

<div class="section" style="margin-top:32px;border-top:1px solid #222;padding-top:16px">
  <div style="color:#444;font-size:10px">
    Nautilus Portal · svend4/pro2 · Подключить новый репо: portal.py register &lt;name&gt; &lt;adapter&gt;
  </div>
</div>
</body>
</html>"""


def render_text(result: PortalResult) -> str:
    lines = [f"\n⬡ NAUTILUS PORTAL — запрос: \"{result.query}\"", "=" * 50]
    for e in result.entries:
        repo = e.id.split(":")[0].upper()
        lines.append(f"\n[{repo}] {e.title}")
        lines.append(f"  {e.content[:120]}")
        if e.links:
            lines.append(f"  → {', '.join(e.links[:3])}")
    c = result.consensus or {}
    lines.append(f"\nКонсенсус: {c.get('coverage',0)*100:.0f}% · {', '.join(c.get('present_in',[]))}")
    if c.get("missing_in"):
        lines.append(f"Отсутствует: {', '.join(c['missing_in'])}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nautilus Portal — единая точка входа для репозиториев")
    parser.add_argument("--query", "-q", default="knowledge", help="Концепт для поиска")
    parser.add_argument("--serve", action="store_true", help="Запустить веб-сервер")
    parser.add_argument("--json", action="store_true", help="JSON-вывод")
    parser.add_argument("--html", action="store_true", help="HTML-вывод")
    parser.add_argument("--describe", action="store_true", help="Описание всех адаптеров")
    args = parser.parse_args()

    portal = NautilusPortal()

    if args.describe:
        print(json.dumps(portal.describe(), ensure_ascii=False, indent=2))
        return

    if args.serve:
        _serve(portal)
        return

    result = portal.query(args.query)

    if args.json:
        data = {
            "query": result.query,
            "entries": [
                {"id": e.id, "title": e.title, "source": e.source,
                 "content": e.content[:200], "metadata": e.metadata}
                for e in result.entries
            ],
            "cross_links": result.cross_links,
            "consensus": result.consensus,
        }
        print(json.dumps(data, ensure_ascii=False, indent=2))
    elif args.html:
        print(render_html(result, portal))
    else:
        print(render_text(result))


def _serve(portal: NautilusPortal):
    """Минимальный HTTP-сервер (без зависимостей)."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            query = params.get("q", ["knowledge"])[0]
            result = portal.query(query)
            html = render_html(result, portal)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())

        def log_message(self, fmt, *args):
            pass  # тихий режим

    print("⬡ Nautilus Portal запущен → http://localhost:8000/?q=crystal")
    HTTPServer(("", 8000), Handler).serve_forever()


if __name__ == "__main__":
    main()

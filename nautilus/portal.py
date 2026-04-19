"""
Nautilus Portal — движок портала для nautilus/ репо.
Использует адаптеры из adapters/ вместо монолитного portal.py.
"""

import json
import sys
import argparse
from dataclasses import dataclass
from typing import Optional

from adapters import (
    Info1Adapter, Pro2Adapter, MetaAdapter, Data2Adapter, Data7Adapter,
    InfoSystemsAdapter, AIAgentsAdapter,
)
from adapters.base import PortalEntry


@dataclass
class PortalResult:
    query: str
    entries: list
    cross_links: list
    consensus: Optional[dict] = None


class NautilusPortal:
    def __init__(self):
        self.adapters = {
            "info1":       Info1Adapter(),
            "pro2":        Pro2Adapter(),
            "meta":        MetaAdapter(),
            "data2":       Data2Adapter(),
            "data7":       Data7Adapter(),
            "infosystems": InfoSystemsAdapter(),
            "ai_agents":   AIAgentsAdapter(),
        }

    def query(self, concept: str) -> PortalResult:
        all_entries = []
        for adapter in self.adapters.values():
            all_entries.extend(adapter.fetch(concept))
        return PortalResult(
            query=concept,
            entries=all_entries,
            cross_links=self._cross_links(all_entries),
            consensus=self._consensus(all_entries),
        )

    def describe(self) -> dict:
        return {name: a.describe() for name, a in self.adapters.items()}

    def register(self, name: str, adapter) -> None:
        self.adapters[name] = adapter

    def _cross_links(self, entries: list) -> list:
        links, seen = [], set()
        for e in entries:
            for link_id in e.links:
                key = tuple(sorted([e.id, link_id]))
                if key not in seen:
                    seen.add(key)
                    sr = e.id.split(":")[0]
                    tr = link_id.split(":")[0]
                    if sr != tr:
                        links.append({"from": e.id, "to": link_id,
                                      "from_repo": sr, "to_repo": tr})
        return links

    def _consensus(self, entries: list) -> dict:
        sources = {e.id.split(":")[0] for e in entries}
        present = sources & set(self.adapters)
        cov = len(present) / len(self.adapters)
        return {
            "present_in": sorted(present),
            "missing_in": sorted(set(self.adapters) - present),
            "coverage": round(cov, 2),
            "agreed": cov >= 1.0,
        }


def render_html(result: PortalResult, portal: "NautilusPortal") -> str:
    c = result.consensus or {}
    coverage_pct = int(c.get("coverage", 0) * 100)
    present = ", ".join(c.get("present_in", []))
    missing = ", ".join(c.get("missing_in", []))

    entries_html = ""
    for e in result.entries:
        repo = e.id.split(":")[0].upper()
        links_html = ""
        if e.links:
            links_html = "<div class='links'>→ " + ", ".join(
                f"<code>{l}</code>" for l in e.links[:4]
            ) + "</div>"
        entries_html += f"""
        <div class="entry">
            <div class="repo-tag">{repo}</div>
            <div class="title">{e.title}</div>
            <div class="content">{e.content[:200]}</div>
            {links_html}
        </div>"""

    cross_html = ""
    for lnk in result.cross_links[:10]:
        cross_html += (
            f"<li><code>{lnk['from']}</code> → <code>{lnk['to']}</code> "
            f"<span class='repos'>({lnk['from_repo']} ↔ {lnk['to_repo']})</span></li>"
        )

    adapters_list = " · ".join(portal.adapters.keys())

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>⬡ Nautilus Portal — {result.query}</title>
<style>
  body {{ font-family: monospace; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
  h1 {{ color: #58a6ff; }}
  .query-form {{ margin: 16px 0; }}
  .query-form input {{ background: #161b22; color: #c9d1d9; border: 1px solid #30363d;
                       padding: 8px 12px; font-size: 16px; width: 300px; border-radius: 6px; }}
  .query-form button {{ background: #238636; color: #fff; border: none; padding: 8px 16px;
                        font-size: 16px; border-radius: 6px; cursor: pointer; margin-left: 8px; }}
  .consensus {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 12px 16px; margin: 16px 0; }}
  .coverage {{ font-size: 24px; color: {'#3fb950' if coverage_pct == 100 else '#d29922' if coverage_pct > 50 else '#f85149'}; }}
  .entry {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
            padding: 12px 16px; margin: 10px 0; }}
  .repo-tag {{ display: inline-block; background: #1f6feb; color: #fff; padding: 2px 8px;
               border-radius: 4px; font-size: 12px; margin-bottom: 6px; }}
  .title {{ font-size: 16px; font-weight: bold; color: #e6edf3; margin-bottom: 4px; }}
  .content {{ color: #8b949e; font-size: 13px; white-space: pre-wrap; }}
  .links {{ color: #58a6ff; font-size: 12px; margin-top: 6px; }}
  .cross-links {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                  padding: 12px 16px; margin: 16px 0; }}
  .cross-links li {{ margin: 4px 0; font-size: 13px; }}
  .repos {{ color: #8b949e; }}
  code {{ background: #21262d; padding: 1px 4px; border-radius: 3px; font-size: 12px; }}
  .adapters {{ color: #8b949e; font-size: 12px; margin-top: 24px; }}
</style>
</head>
<body>
<h1>⬡ Nautilus Portal</h1>
<form class="query-form" method="get">
  <input name="q" value="{result.query}" placeholder="запрос...">
  <button type="submit">Найти</button>
</form>

<div class="consensus">
  <span class="coverage">{coverage_pct}% покрытие</span> &nbsp;·&nbsp;
  найдено в: <strong>{present or '—'}</strong>
  {f'&nbsp;·&nbsp; отсутствует: {missing}' if missing else ''}
</div>

<div>{entries_html}</div>

{'<div class="cross-links"><strong>Межрепозиторные связи:</strong><ul>' + cross_html + '</ul></div>' if cross_html else ''}

<div class="adapters">адаптеры: {adapters_list}</div>
</body>
</html>"""


def render_text(result: PortalResult) -> str:
    lines = [f'\n⬡ NAUTILUS PORTAL — "{result.query}"', "=" * 50]
    for e in result.entries:
        repo = e.id.split(":")[0].upper()
        lines.append(f"\n[{repo}] {e.title}")
        lines.append(f"  {e.content[:120]}")
        if e.links:
            lines.append(f"  → {', '.join(e.links[:3])}")
    c = result.consensus or {}
    lines.append(f"\nКонсенсус: {c.get('coverage',0)*100:.0f}% · "
                 f"{', '.join(c.get('present_in',[]))}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", default="knowledge")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--describe", action="store_true")
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
        print(json.dumps({
            "query": result.query,
            "entries": [{"id": e.id, "title": e.title, "content": e.content[:200],
                         "metadata": e.metadata} for e in result.entries],
            "cross_links": result.cross_links,
            "consensus": result.consensus,
        }, ensure_ascii=False, indent=2))
    else:
        print(render_text(result))


def _serve(portal: NautilusPortal):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            query = params.get("q", ["knowledge"])[0]
            result = portal.query(query)
            html = render_html(result, portal)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())

        def log_message(self, *a):
            pass

    print("⬡ Nautilus Portal → http://localhost:8000/?q=crystal")
    HTTPServer(("", 8000), Handler).serve_forever()


if __name__ == "__main__":
    main()

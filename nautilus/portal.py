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
    # импортируем рендерер из основного portal.py (один уровень выше)
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    from portal import render_html

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

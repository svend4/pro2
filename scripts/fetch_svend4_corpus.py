"""
Загрузчик обучающего корпуса из репозиториев svend4.

Категории:
  1. ai_agents    — ИИ и агенты        (info4, info5, data30, daten, info7)
  2. infosystems  — Инфо-системы       (info, daten11, data10, info1)
  3. knowledge    — Знания/архетипы    (info3, daten22)
  4. algorithms   — Алгоритмы/опт.     (data7)

Использование:
    python scripts/fetch_svend4_corpus.py --output data/svend4_corpus
    python scripts/fetch_svend4_corpus.py --output data/svend4_corpus --stats
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
import base64


# ─────────────────────────────────────────────
#  Конфигурация репозиториев
# ─────────────────────────────────────────────

REPOS = {
    "ai_agents": [
        "info4",   # AI-скиллы, псевдо RAG, ~1.75 MB
        "info5",   # 4-уровневая пирамида автоматизации, ~310 KB
        "data30",  # ОС рационал программ, ~720 KB
        "daten",   # Информационная ОС, ~150 KB
        "info7",   # Оркестратор+бот, ~4 MB общий
    ],
    "infosystems": [
        "info",     # Энциклопедия, параллели, ~738 KB
        "daten11",  # Метаданные 4 уровня, ~138 KB
        "data10",   # Dynamic Content Blocks, ~193 KB
        "info1",    # Инфо-система, шаблоны, алгоритмы
    ],
    "knowledge": [
        "info3",   # Гуманитарные формулы, архетипы, ~18 KB
        "daten22", # SQLite FTS4, 16 архетипов, псевдо RAG
    ],
    "algorithms": [
        "data7",   # TSP, MMORPG, энциклопедии, ~2 MB
    ],
}

TEXT_EXTENSIONS = {".md", ".txt", ".skill", ".rst"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".github", "vendor"}

GITHUB_API = "https://api.github.com"
USER = "svend4"
RATE_LIMIT_PAUSE = 2.0  # сек между запросами при ошибках


# ─────────────────────────────────────────────
#  Утилиты GitHub API
# ─────────────────────────────────────────────

def _get(url: str, retries: int = 4) -> dict | list | None:
    """GET-запрос к GitHub API с экспоненциальным backoff."""
    wait = 2
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 403:  # rate limit
                print(f"  [rate-limit] ожидаем {wait}s …", flush=True)
                time.sleep(wait)
                wait *= 2
            elif e.code == 404:
                return None
            else:
                print(f"  [HTTP {e.code}] {url}", flush=True)
                return None
        except Exception as exc:
            print(f"  [error] {exc} — попытка {attempt+1}/{retries}", flush=True)
            time.sleep(wait)
            wait *= 2
    return None


def fetch_tree(repo: str) -> list[dict]:
    """Возвращает полное дерево файлов репозитория."""
    url = f"{GITHUB_API}/repos/{USER}/{repo}/git/trees/HEAD?recursive=1"
    data = _get(url)
    if data is None:
        return []
    return [item for item in data.get("tree", []) if item.get("type") == "blob"]


def fetch_file_content(repo: str, path: str) -> str | None:
    """Загружает содержимое файла через Contents API."""
    url = f"{GITHUB_API}/repos/{USER}/{repo}/contents/{path}"
    data = _get(url)
    if data is None or "content" not in data:
        return None
    try:
        raw = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return raw
    except Exception:
        return None


# ─────────────────────────────────────────────
#  Фильтрация файлов
# ─────────────────────────────────────────────

def is_text_file(path: str) -> bool:
    """Проверяет, является ли файл текстовым по расширению."""
    _, ext = os.path.splitext(path.lower())
    return ext in TEXT_EXTENSIONS


def is_in_skip_dir(path: str) -> bool:
    parts = path.split("/")
    return any(p in SKIP_DIRS for p in parts)


# ─────────────────────────────────────────────
#  Основная логика скачивания
# ─────────────────────────────────────────────

def fetch_repo(repo: str, out_dir: str) -> dict:
    """
    Скачивает все текстовые файлы репозитория в out_dir.
    Возвращает статистику.
    """
    os.makedirs(out_dir, exist_ok=True)
    stats = {"repo": repo, "files": 0, "bytes": 0, "skipped": 0}

    print(f"  [{repo}] получаем дерево файлов …", flush=True)
    tree = fetch_tree(repo)

    if not tree:
        print(f"  [{repo}] дерево пустое или ошибка", flush=True)
        return stats

    text_files = [
        item for item in tree
        if is_text_file(item["path"]) and not is_in_skip_dir(item["path"])
    ]
    print(f"  [{repo}] найдено {len(text_files)} текстовых файлов", flush=True)

    for item in text_files:
        path = item["path"]
        content = fetch_file_content(repo, path)
        if content is None:
            stats["skipped"] += 1
            continue

        # Сохраняем с сохранением структуры: repo/original/path.md
        local_path = os.path.join(out_dir, path.replace("/", "__"))
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(f"# SOURCE: {USER}/{repo}/{path}\n\n")
            f.write(content)

        stats["files"] += 1
        stats["bytes"] += len(content.encode("utf-8"))

        # Небольшая пауза чтобы не бить по rate limit
        time.sleep(0.3)

    print(f"  [{repo}] скачано {stats['files']} файлов, {stats['bytes']//1024} KB", flush=True)
    return stats


def build_corpus(output_dir: str) -> dict:
    """Скачивает весь корпус по всем категориям."""
    all_stats = {}

    for category, repos in REPOS.items():
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        print(f"\n{'='*50}", flush=True)
        print(f"Категория: {category}", flush=True)
        print(f"{'='*50}", flush=True)

        cat_stats = {"repos": [], "total_files": 0, "total_bytes": 0}
        for repo in repos:
            repo_dir = os.path.join(cat_dir, repo)
            stats = fetch_repo(repo, repo_dir)
            cat_stats["repos"].append(stats)
            cat_stats["total_files"] += stats["files"]
            cat_stats["total_bytes"] += stats["bytes"]

        all_stats[category] = cat_stats

    return all_stats


def print_summary(all_stats: dict):
    """Выводит итоговую статистику."""
    print(f"\n{'='*60}")
    print("ИТОГО ПО КОРПУСУ")
    print(f"{'='*60}")

    grand_files = 0
    grand_bytes = 0

    for category, cat_stats in all_stats.items():
        files = cat_stats["total_files"]
        kb = cat_stats["total_bytes"] // 1024
        grand_files += files
        grand_bytes += cat_stats["total_bytes"]
        print(f"  {category:15s}: {files:4d} файлов  {kb:6d} KB")

        for r in cat_stats["repos"]:
            print(f"    └─ {r['repo']:12s}: {r['files']:3d} файлов  {r['bytes']//1024:5d} KB")

    print(f"{'─'*60}")
    print(f"  {'ВСЕГО':15s}: {grand_files:4d} файлов  {grand_bytes//1024:6d} KB  ({grand_bytes/1024/1024:.1f} MB)")


def save_manifest(all_stats: dict, output_dir: str):
    """Сохраняет манифест корпуса."""
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\nМанифест: {manifest_path}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Загрузчик корпуса svend4 для обучения")
    parser.add_argument("--output", default="data/svend4_corpus", help="Директория вывода")
    parser.add_argument("--stats", action="store_true", help="Только статистика без скачивания")
    parser.add_argument("--category", default=None,
                        help="Скачать только одну категорию (ai_agents|infosystems|knowledge|algorithms)")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    if args.stats:
        # Только считаем размеры через API
        print("Режим статистики (без скачивания)\n")
        for category, repos in REPOS.items():
            if args.category and category != args.category:
                continue
            print(f"Категория: {category}")
            for repo in repos:
                tree = fetch_tree(repo)
                text_files = [f for f in tree if is_text_file(f["path"]) and not is_in_skip_dir(f["path"])]
                total_size = sum(f.get("size", 0) for f in text_files)
                print(f"  {repo:15s}: {len(text_files):3d} файлов  ~{total_size//1024:5d} KB")
        return

    if args.category:
        if args.category not in REPOS:
            print(f"Неизвестная категория: {args.category}")
            print(f"Доступные: {list(REPOS.keys())}")
            sys.exit(1)
        repos_to_fetch = {args.category: REPOS[args.category]}
    else:
        repos_to_fetch = REPOS

    # Временно подменяем REPOS для частичной загрузки
    global REPOS
    original = REPOS
    REPOS = repos_to_fetch

    print(f"Загружаем корпус в: {output_dir}")
    all_stats = build_corpus(output_dir)
    print_summary(all_stats)
    save_manifest(all_stats, output_dir)

    REPOS = original


if __name__ == "__main__":
    main()

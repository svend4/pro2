"""
Загрузчик обучающего корпуса из репозиториев svend4.

Использует git clone (быстро, без rate-limit ограничений).

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
import shutil
import argparse
import subprocess
import urllib.request
import urllib.error
import urllib.parse


# ─────────────────────────────────────────────
#  Конфигурация репозиториев
# ─────────────────────────────────────────────

REPOS = {
    "ai_agents": [
        "info4",   # AI-скиллы, псевдо RAG, ~3.2 MB
        "info5",   # 4-уровневая пирамида автоматизации, ~261 KB
        "data30",  # ОС рационал программ, ~703 KB
        "daten",   # Информационная ОС, ~432 KB
        "info7",   # Оркестратор+бот, ~2 MB
    ],
    "infosystems": [
        "info",     # Энциклопедия, параллели, ~740 KB
        "daten11",  # Метаданные 4 уровня, ~149 KB
        "data10",   # Dynamic Content Blocks, ~188 KB
        "info1",    # Инфо-система, шаблоны, алгоритмы, ~4 MB
    ],
    "knowledge": [
        "info3",   # Гуманитарные формулы, архетипы, ~4.4 MB
        "daten22", # SQLite FTS4, 16 архетипов, ~1.4 MB
    ],
    "algorithms": [
        "data7",   # TSP, MMORPG, энциклопедии
    ],
}

TEXT_EXTENSIONS = {".md", ".txt", ".skill", ".rst"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".github", "vendor"}

GITHUB_BASE = "https://github.com"
USER = "svend4"


# ─────────────────────────────────────────────
#  Git clone
# ─────────────────────────────────────────────

def clone_repo(repo: str, clone_dir: str) -> bool:
    """Клонирует репозиторий через git (shallow clone, только текст)."""
    url = f"{GITHUB_BASE}/{USER}/{repo}.git"
    if os.path.isdir(os.path.join(clone_dir, ".git")):
        # Уже клонирован — обновляем
        print(f"  [{repo}] обновляем (git pull) …", flush=True)
        result = subprocess.run(
            ["git", "-C", clone_dir, "pull", "--quiet"],
            capture_output=True, timeout=120
        )
        return result.returncode == 0

    print(f"  [{repo}] клонируем {url} …", flush=True)
    os.makedirs(clone_dir, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth=1", "--quiet", url, clone_dir],
        capture_output=True, timeout=180
    )
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        print(f"  [{repo}] ошибка clone: {err}", flush=True)
        return False
    return True


# ─────────────────────────────────────────────
#  Фильтрация и сбор текстов
# ─────────────────────────────────────────────

def is_text_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in TEXT_EXTENSIONS


def is_in_skip_dir(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    return any(p in SKIP_DIRS for p in parts)


def collect_texts(clone_dir: str, out_dir: str, repo: str) -> dict:
    """Копирует текстовые файлы из клонированного репо в out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    stats = {"repo": repo, "files": 0, "bytes": 0}

    for root, dirs, files in os.walk(clone_dir):
        # Пропускаем скрытые/служебные директории
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, clone_dir)
            if not is_text_file(fname) or is_in_skip_dir(rel):
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                # Сохраняем в flat-структуру с источником в заголовке
                safe_name = rel.replace(os.sep, "__")
                dst = os.path.join(out_dir, safe_name)
                with open(dst, "w", encoding="utf-8") as f:
                    f.write(f"# SOURCE: {USER}/{repo}/{rel}\n\n")
                    f.write(content)
                stats["files"] += 1
                stats["bytes"] += len(content.encode("utf-8"))
            except Exception:
                pass

    print(f"  [{repo}] скопировано {stats['files']} файлов, {stats['bytes']//1024} KB", flush=True)
    return stats


# ─────────────────────────────────────────────
#  API stats (только для --stats режима)
# ─────────────────────────────────────────────

def _api_get(url: str) -> dict | None:
    parsed = urllib.parse.urlparse(url)
    encoded_path = urllib.parse.quote(parsed.path, safe="/:@!$&'()*+,;=")
    safe_url = urllib.parse.urlunparse(parsed._replace(path=encoded_path))
    wait = 2
    for _ in range(4):
        try:
            req = urllib.request.Request(safe_url, headers={"Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"  [rate-limit] ожидаем {wait}s …", flush=True)
                time.sleep(wait); wait *= 2
            else:
                return None
        except Exception:
            time.sleep(wait); wait *= 2
    return None


def fetch_tree_stats(repo: str) -> tuple[int, int]:
    """Возвращает (кол-во текстовых файлов, суммарный размер байт) через API."""
    data = _api_get(f"https://api.github.com/repos/{USER}/{repo}/git/trees/HEAD?recursive=1")
    if not data:
        return 0, 0
    items = [i for i in data.get("tree", [])
             if i.get("type") == "blob"
             and is_text_file(i["path"])
             and not is_in_skip_dir(i["path"])]
    return len(items), sum(i.get("size", 0) for i in items)


# ─────────────────────────────────────────────
#  Основная логика
# ─────────────────────────────────────────────

def build_corpus_from(repos_dict: dict, output_dir: str) -> dict:
    """Клонирует репозитории и собирает текстовые файлы по категориям."""
    tmp_dir = os.path.join(output_dir, "_clones")
    all_stats = {}

    for category, repos in repos_dict.items():
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        print(f"\n{'='*50}", flush=True)
        print(f"Категория: {category}", flush=True)
        print(f"{'='*50}", flush=True)

        cat_stats = {"repos": [], "total_files": 0, "total_bytes": 0}
        for repo in repos:
            clone_dir = os.path.join(tmp_dir, repo)
            repo_out_dir = os.path.join(cat_dir, repo)

            ok = clone_repo(repo, clone_dir)
            if not ok:
                print(f"  [{repo}] пропускаем (ошибка клонирования)", flush=True)
                cat_stats["repos"].append({"repo": repo, "files": 0, "bytes": 0})
                continue

            stats = collect_texts(clone_dir, repo_out_dir, repo)
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
        repos_to_fetch = dict(REPOS)

    print(f"Загружаем корпус в: {output_dir}")
    all_stats = build_corpus_from(repos_to_fetch, output_dir)
    print_summary(all_stats)
    save_manifest(all_stats, output_dir)


if __name__ == "__main__":
    main()

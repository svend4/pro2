#!/usr/bin/env python3
"""
monitor_improve.py — Монитор прогресса e2_self_improve в реальном времени.
Запустить в отдельном терминале пока идёт e2_self_improve.py.

Использование:
    python monitor_improve.py
    python monitor_improve.py --watch 10   # обновлять каждые 10 секунд
"""

import json
import time
import argparse
from pathlib import Path

LOG = Path('e2_self_improve_log.json')


def render(log: list):
    """Красиво отобразить текущий прогресс."""
    print("\033[2J\033[H")  # Очистить экран
    print("=" * 60)
    print("  МОНИТОРИНГ e2_self_improve")
    print("=" * 60)

    if not log:
        print("  Лог пустой — ожидаем первую итерацию...")
        return

    ppls = [it['avg_ppl'] for it in log]
    target = log[0].get('target_ppl', 100)

    # График PPL
    print(f"\n  PPL по итерациям (цель={target}):\n")
    for it in log:
        ppl = it['avg_ppl']
        bar_len = int((1 - min(ppl / 250, 1)) * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        arrow = "[OK]" if ppl <= target else "    "
        print(f"  Iter {it['iteration']:2d}: [{bar}] {ppl:7.2f} {arrow}")

    # Улучшения в последней итерации
    last = log[-1]
    print(f"\n  Последняя итерация #{last['iteration']}:")
    for imp in last.get('improvements', []):
        delta = imp.get('ppl_delta', 0)
        sign = "down" if delta > 0 else "up"
        ok = "[OK]" if delta > 0 else "[!!]"
        print(f"    {ok} {imp['label']:<25} "
              f"{imp['ppl_before']:.1f} -> {imp['ppl_after']:.1f} "
              f"({sign} {abs(delta):.1f}%)")

    # Тренд
    if len(ppls) > 1:
        trend = ppls[-1] - ppls[0]
        direction = "улучшение" if trend < 0 else "ухудшение"
        print(f"\n  Тренд: {direction} на {abs(trend):.2f} PPL")
        print(f"  Прогресс: {ppls[0]:.2f} -> {ppls[-1]:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', type=int, default=5, help='Интервал обновления (сек)')
    parser.add_argument('--once', action='store_true', help='Показать один раз и выйти')
    parser.add_argument('--log', default='', help='Путь к лог-файлу')
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else LOG

    while True:
        try:
            if log_path.exists():
                log = json.loads(log_path.read_text())
                render(log)
            else:
                print(f"Ожидаем {log_path}...")

            if args.once:
                break
            time.sleep(args.watch)

        except KeyboardInterrupt:
            print("\nМониторинг остановлен.")
            break
        except Exception as e:
            print(f"Ошибка: {e}")
            time.sleep(args.watch)


if __name__ == '__main__':
    main()

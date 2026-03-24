#!/bin/bash
# run_all_checks.sh — Полная проверка состояния проекта.
# Запускать после каждого значимого изменения.

set -e
cd "$(dirname "$0")"
mkdir -p experiments

CHECKPOINT="${1:-hmoe_self_trained_v4.pt}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')

echo ""
echo "======================================================"
echo "  ПОЛНАЯ ПРОВЕРКА ПРОЕКТА"
echo "  $TIMESTAMP"
echo "  Checkpoint: $CHECKPOINT"
echo "======================================================"

# 1. Q4⊂Q6 валидация
echo ""
echo ">>> 1/5 Валидация Q4⊂Q6..."
if python experiments/validate_q4_q6.py 2>&1; then
    Q4_OK="[OK]"
else
    Q4_OK="[FAIL]"
fi

# 2. Тест per-source trit_proj
echo ""
echo ">>> 2/5 Тест ArchetypalInterlinguaFixed..."
if python yijing_transformer/models/geometry/interlingua_fixed.py 2>&1; then
    INTER_OK="[OK]"
else
    INTER_OK="[FAIL]"
fi

# 3. Тест KasatkinQ6Router
echo ""
echo ">>> 3/5 Тест KasatkinQ6Router..."
if python yijing_transformer/models/geometry/kasatkin_router.py 2>&1; then
    ROUTER_OK="[OK]"
else
    ROUTER_OK="[FAIL]"
fi

# 4. Ксерокс-тест
echo ""
echo ">>> 4/5 Ксерокс-тест..."
if python experiments/xerox_test.py --mock 2>&1; then
    XEROX_OK="[OK]"
else
    XEROX_OK="[FAIL]"
fi

# 5. Сравнение GlyphTokenizer (если файл существует)
echo ""
echo ">>> 5/5 Glyph vs Char comparison (fast)..."
if [ -f "experiments/train_with_glyph.py" ]; then
    if python experiments/train_with_glyph.py --steps 100 --fast 2>&1; then
        GLYPH_OK="[OK]"
    else
        GLYPH_OK="[FAIL]"
    fi
else
    GLYPH_OK="[SKIP]"
    echo "  experiments/train_with_glyph.py не найден — пропуск"
fi

# Итоговый отчёт
echo ""
echo "======================================================"
echo "  ИТОГ ПРОВЕРОК  --  $TIMESTAMP"
echo "======================================================"
echo "  $Q4_OK    1. Q4⊂Q6 математическая верификация"
echo "  $INTER_OK  2. Per-source trit_proj (Interlingua fix)"
echo "  $ROUTER_OK 3. KasatkinQ6Router (Q6-маршрутизация)"
echo "  $XEROX_OK  4. Ксерокс-тест (само-осознание)"
echo "  $GLYPH_OK  5. GlyphTokenizer vs CharTokenizer"
echo "======================================================"
echo "  Результаты в: experiments/"
echo "  Следующий шаг:"
echo "    python e2_self_improve.py --iters 3 --target-ppl 100 --no-v3"
echo "======================================================"

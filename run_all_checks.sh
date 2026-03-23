#!/bin/bash
# run_all_checks.sh — запустить все проверки и получить сводку

set -e
cd "$(dirname "$0")"

echo ""
echo "======================================================="
echo "  ПОЛНАЯ ПРОВЕРКА АРХИТЕКТУРЫ"
echo "  Дата: $(date '+%Y-%m-%d %H:%M')"
echo "======================================================="

# 1. Q4⊂Q6 валидация
echo ""
echo ">>> 1/4 Валидация Q4⊂Q6..."
python experiments/validate_q4_q6.py
echo "Готово."

# 2. Тест исправленной Interlingua
echo ""
echo ">>> 2/4 Тест per-source trit_proj..."
python tests/test_interlingua_fixed.py
echo "Готово."

# 3. Тест Gumbel-Softmax квантизатора
echo ""
echo ">>> 3/4 Тест TernaryQuantizerFixed..."
python yijing_transformer/models/geometry/quantizer_fixed.py --test
echo "Готово."

# 4. Ксерокс-тест
echo ""
echo ">>> 4/4 Ксерокс-тест (mock)..."
python experiments/xerox_test.py --mock
echo "Готово."

echo ""
echo "======================================================="
echo "  ВСЕ ПРОВЕРКИ ЗАВЕРШЕНЫ"
echo "  Результаты в: experiments/*.json"
echo "======================================================="

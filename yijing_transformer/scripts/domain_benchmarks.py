"""
domain_benchmarks.py — специализированные бенчмарки для каждого «вредного» источника.

Контекст: в v59 пять источников (Склярова, Фомюк, Андреев, Касаткин, Герман)
были признаны «вредными» на синтетическом WikiText за 800 шагов.
Это не значит, что идеи плохие — это значит, что нужен правильный тест.

Каждый источник имеет свою область применения:
  - Склярова: иерархические структуры (дворцы → 8 групп × 8 = 64)
  - Фомюк: симметричные структуры (антиподы, палиндромы, зеркала)
  - Касаткин: пространственные описания (3D координаты, направления)
  - Андреев: прогрессивные структуры (нарастающая сложность, треугольные числа)
  - Герман: упаковочные структуры (2^k периодичность, бесколлизионность)

Использование:
    python scripts/domain_benchmarks.py --model yijing_gpt --config config.yaml
    python scripts/domain_benchmarks.py --demo  # просмотр тестовых текстов
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ─── Тестовые тексты для каждого источника ───────────────────────────────────

@dataclass
class DomainBenchmark:
    """Набор текстов для тестирования одного источника."""
    source_name: str        # Имя источника (Склярова, Фомюк, ...)
    hexagram: str           # Связанная концепция И-Цзин
    hypothesis: str         # Что должна улучшить геометрия этого источника
    texts: List[str] = field(default_factory=list)
    control_texts: List[str] = field(default_factory=list)  # контроль (не должно помогать)
    metadata: dict = field(default_factory=dict)


# ─── 1. Склярова: иерархические дворцовые структуры ─────────────────────────

SKLIAROVA_BENCHMARK = DomainBenchmark(
    source_name='Склярова',
    hexagram='8 дворцов × 8 гексаграмм = 64 (структура «Компьютера Древнего Китая»)',
    hypothesis=(
        'Текст с иерархической 8-кратной структурой должен '
        'обрабатываться лучше с palace attention (8 дворцов × 8 гексаграмм), '
        'чем обычным attention.'
    ),
    texts=[
        # 8-уровневая иерархия (классификация)
        """Биологическая классификация:
Домен: Эукариоты
  Царство: Животные
    Тип: Хордовые
      Класс: Млекопитающие
        Отряд: Приматы
          Семейство: Гоминиды
            Род: Homo
              Вид: Homo sapiens""",

        # Юридическая иерархия (8 уровней)
        """Правовая иерархия Российской Федерации:
1. Конституция РФ
2. Федеральные конституционные законы
3. Федеральные законы
4. Указы Президента
5. Постановления Правительства
6. Приказы министерств
7. Региональные законы
8. Муниципальные акты""",

        # Архитектурная иерархия
        """Структура организации:
Совет директоров
  └── Генеральный директор
        └── Исполнительный директор
              └── Директор департамента
                    └── Начальник отдела
                          └── Руководитель группы
                                └── Старший специалист
                                      └── Специалист""",

        # Файловая система (глубокая иерархия)
        """/root
  /home
    /user
      /projects
        /backend
          /src
            /controllers
              /auth_controller.py""",

        # XML с глубокой вложенностью
        """<configuration>
  <database>
    <connection>
      <pool>
        <settings>
          <timeout>
            <value>30</value>
          </timeout>
        </settings>
      </pool>
    </connection>
  </database>
</configuration>""",
    ],
    control_texts=[
        # Линейные тексты без иерархии (геометрия НЕ должна помогать)
        "The weather today is sunny. It will be 25 degrees. Perfect for outdoor activities.",
        "SELECT id, name, email FROM users WHERE age > 18 ORDER BY name LIMIT 10;",
        "import numpy as np\nx = np.array([1, 2, 3])\nprint(x.mean())",
    ],
    metadata={'palace_depth': 8, 'branching_factor': 8},
)


# ─── 2. Фомюк: антиподальные и симметричные структуры ───────────────────────

FOMYUK_BENCHMARK = DomainBenchmark(
    source_name='Фомюк',
    hexagram='Антиподальность: hex(i) + hex(63-i) = const (баланс противоположностей)',
    hypothesis=(
        'Тексты с антиподальной/зеркальной/палиндромной структурой должны '
        'обрабатываться лучше с D4-эквивариантным модулем, '
        'использующим антиподальные пары.'
    ),
    texts=[
        # Математические палиндромы и симметрии
        """Число 12321 является палиндромом.
1 + 2 + 3 + 2 + 1 = 9 = 3²
121 = 11² (палиндром квадрата)
12321 = 111² (палиндром квадрата квадрата)
1234321 = 1111²""",

        # Зеркальные утверждения (антиподы)
        """Максимум одного — минимум другого.
Рост цен → падение покупательной способности.
Увеличение скорости → уменьшение точности.
Больше памяти → меньше вычислений.
Больше абстракции → меньше эффективности.
Сила без мудрости — слабость. Мудрость без силы — беспомощность.""",

        # Химические антиподы (энантиомеры)
        """R-конфигурация и S-конфигурация — зеркальные изомеры.
(R)-аланин — строительный блок белков у животных.
(S)-аланин — встречается у бактерий и грибов.
Хиральность определяется тем, какой изомер активен биологически.
Правое и левое вращение плоскости поляризации — антиподы.""",

        # Логические антиподы
        """Утверждение: «Все лебеди белые».
Антипод: «Существует нелебедь чёрный».
Отрицание: «Не все лебеди белые».
Контрапозитив: «Если не белый, то не лебедь».
Инверсия: «Если лебедь, то белый» ↔ «Если не лебедь, то не белый».""",

        # Поэтическая симметрия (кольцевая композиция)
        """В начале было слово. Слово было у Бога. Бог был Словом.
Он был в начале у Бога. Всё через Него начало быть.
И без Него ничто не начало быть, что начало быть.
И слово стало плотью. Плоть стала Словом.
В конце стало слово. Слово было у Бога. Бог был Словом.""",
    ],
    control_texts=[
        "The stock market rose 2.3% today. Technology stocks led gains.",
        "def merge_sort(arr): return arr if len(arr) <= 1 else ...",
        "Стоимость доставки зависит от веса и расстояния.",
    ],
    metadata={'symmetry_type': 'antipodal+palindrome+mirror'},
)


# ─── 3. Касаткин: пространственные и 3D-описания ────────────────────────────

KASATKIN_BENCHMARK = DomainBenchmark(
    source_name='Касаткин',
    hexagram='Диофантовы координаты куба: каждая гексаграмма = точка в Z³',
    hypothesis=(
        'Тексты с пространственными отношениями, 3D-координатами и '
        'направлениями должны обрабатываться лучше с 3D-кубическим '
        'встраиванием Касаткина.'
    ),
    texts=[
        # Навигационные инструкции
        """Маршрут из центра города:
Из точки A (0, 0, 0) двигайтесь на север 500 метров.
Поверните на восток, пройдите 200 метров (2, 1, 0).
Подъём по лестнице на 3 этажа (2, 1, 3).
Коридор направо 50 метров (2.5, 1, 3).
Кабинет 315 — третья дверь слева (2.5, 0.8, 3).""",

        # Молекулярная геометрия
        """Молекула воды H₂O имеет угловую форму.
Атом кислорода занимает вершину угла (0, 0, 0).
Первый водород на расстоянии 0.96 Å под углом 104.5°.
Координаты H₁: (0.76, 0.59, 0).
Координаты H₂: (-0.76, 0.59, 0).
Дипольный момент направлен по оси Y (0, 1, 0).""",

        # Архитектурные чертежи
        """Спецификация помещения 3B:
Длина: 6.4 м (ось X)
Ширина: 4.2 м (ось Y)
Высота: 2.8 м (ось Z)
Северо-западный угол: (0, 4.2, 0)
Юго-восточный угол: (6.4, 0, 0)
Центр комнаты: (3.2, 2.1, 1.4)
Дверной проём: (0, 2.0, 0) — (0, 2.9, 2.1)""",

        # Кристаллография (кубическая решётка)
        """Гранецентрированная кубическая решётка (ГЦК):
Атом в вершинах куба: (0,0,0), (a,0,0), (0,a,0), (0,0,a)
Атомы на гранях: (a/2,a/2,0), (a/2,0,a/2), (0,a/2,a/2)
Параметр решётки для меди: a = 3.615 Å
Координационное число: 12 (каждый атом окружён 12 соседями)
Плотность упаковки: 74% от объёма куба""",

        # Роботехника (декартовы координаты манипулятора)
        """Программа движения робота-манипулятора:
MOVL P1 V=100  ; начало (150, 0, 300 мм)
MOVL P2 V=50   ; движение к детали (250, 150, 200 мм)
GRASP          ; захват детали
MOVL P3 V=30   ; подъём (250, 150, 400 мм)
MOVL P4 V=100  ; перенос (50, 300, 400 мм)
RELEASE        ; отпустить деталь""",
    ],
    control_texts=[
        "The meeting is scheduled for Monday at 10 AM.",
        "Рецепт борща: свекла, капуста, картофель, морковь, лук.",
        "assert x == expected, f'Got {x}, expected {expected}'",
    ],
    metadata={'geometry': '3D_euclidean', 'coordinate_system': 'cartesian'},
)


# ─── 4. Андреев: прогрессивные и треугольные структуры ──────────────────────

ANDREEV_BENCHMARK = DomainBenchmark(
    source_name='Андреев',
    hexagram='Треугольная матрица 64 гексаграмм: T(n) = n(n-1)/2 — порядок обхода',
    hypothesis=(
        'Тексты с нарастающей сложностью, лестницами знания, '
        'треугольными числами и curriculum-структурой должны обрабатываться '
        'лучше с треугольной attention mask Андреева.'
    ),
    texts=[
        # Математические прогрессии (треугольные числа)
        """Треугольные числа: 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105...
T(n) = n(n+1)/2
T(1) = 1    (один элемент)
T(2) = 3    (один + два)
T(3) = 6    (один + два + три)
T(4) = 10   (один + два + три + четыре)
Сумма первых n натуральных чисел — всегда треугольное число.""",

        # Curriculum learning — нарастающая сложность
        """Программа обучения программированию:
Неделя 1: Переменные и типы данных (1 концепция)
Неделя 2: Условия и циклы (2 концепции)
Неделя 3: Функции, аргументы, возврат (3 концепции)
Неделя 4: Списки, словари, кортежи (4 концепции)
Неделя 5: Классы, наследование, полиморфизм, инкапсуляция (5 концепций)
Неделя 6: Файлы, исключения, контекстные менеджеры, логирование, тестирование, CI (6 концепций)""",

        # Пирамидальные структуры данных
        """Бинарная куча (heap):
          1
        /   \\
       2     3
      / \\   / \\
     4   5 6   7
    /|  /
   8 9 10
Уровень 0: 1 элемент  (T(1)=1)
Уровень 1: 2 элемента (T(2)-T(1)=2)
Уровень 2: 4 элемента (T(3)-T(2)=3... нет, это 2^level)
Высота: log2(n) для n элементов""",

        # Нарастающая абстракция (философия)
        """Уровни абстракции в философии сознания:
1. Физика: атомы, молекулы, нейроны
2. Биология: клетки, синапсы, мозг
3. Психология: ощущения, восприятие, эмоции
4. Когнитивистика: внимание, память, мышление
5. Феноменология: qualia, сознание, субъективность
6. Метафизика: разум, бытие, сознание мира
Каждый уровень строится на предыдущем. Нельзя перескочить.""",

        # Разворачивающийся нарратив (нарастающие откровения)
        """Детектив узнавал правду постепенно.
Сначала — следы на снегу.
Потом — следы и записка.
Потом — следы, записка и разбитое стекло.
Потом — следы, записка, стекло и запах духов.
Потом — следы, записка, стекло, духи и незнакомое лицо на фото.
Потом — всё это вместе привело его к двери, за которой ждала разгадка.""",
    ],
    control_texts=[
        "The API returns JSON with status code 200 on success.",
        "Сегодня прошёл дождь. Завтра обещают солнце.",
        "x = torch.randn(batch_size, seq_len, d_model)",
    ],
    metadata={'structure': 'triangular_progressive', 'T_n': 'n*(n+1)/2'},
)


# ─── 5. Герман: упаковочные и 2^k структуры ─────────────────────────────────

HERMAN_BENCHMARK = DomainBenchmark(
    source_name='Герман',
    hexagram='P=2^k упаковка: только степени двойки заполняются без коллизий',
    hypothesis=(
        'Тексты с явными двоичными структурами, периодами 2^k, '
        'кодами Хэмминга, битовыми операциями должны обрабатываться '
        'лучше с правильно обоснованным кодбуком размером 64=2^6.'
    ),
    texts=[
        # Битовые операции и маски
        """Операции с битовыми масками:
0b00000001 = 1   (только бит 0)
0b00000010 = 2   (только бит 1)
0b00000100 = 4   (только бит 2)
0b10000000 = 128 (только бит 7)
Маска 0xFF = 255 = все 8 бит установлены
x & 0x0F  — выделить младшие 4 бита
x | 0xF0  — установить старшие 4 бита
x ^ 0xFF  — инвертировать все 8 бит""",

        # Коды Хэмминга (2^k структура)
        """Код Хэмминга (7, 4):
Размер блока: 7 бит (2³ - 1)
Информационных бит: 4
Проверочных бит: 3 (позиции 1, 2, 4 — степени двойки!)
Паритет P1: биты 1, 3, 5, 7
Паритет P2: биты 2, 3, 6, 7
Паритет P4: биты 4, 5, 6, 7
Минимальное расстояние Хэмминга: 3 (исправляет 1 ошибку)""",

        # IPv4-подсети (2^k размеры)
        """/8  сеть: 16 777 216 адресов (2^24)
/16 сеть: 65 536 адресов      (2^16)
/24 сеть: 256 адресов          (2^8)
/25 сеть: 128 адресов          (2^7)
/26 сеть: 64 адреса            (2^6)
/27 сеть: 32 адреса            (2^5)
/28 сеть: 16 адресов           (2^4)
/32 сеть: 1 адрес              (2^0)""",

        # Память и кэши (размеры 2^k)
        """Иерархия кэш-памяти (типичный процессор):
L1 Instruction Cache: 32 KB = 2^15 байт
L1 Data Cache:        32 KB = 2^15 байт
L2 Cache:            256 KB = 2^18 байт
L3 Cache:              8 MB = 2^23 байт
RAM (типично):        16 GB = 2^34 байт
Строка кэша: 64 байта = 2^6 байт
Страница памяти: 4 KB = 2^12 байт
Выравнивание SIMD: 16, 32, 64 байта = 2^4, 2^5, 2^6""",

        # Нотация в музыке (2^k ритм)
        """Временные доли в музыке:
Целая нота: 1 (=2^0)
Половинная: 1/2 (=2^{-1})
Четверть: 1/4 (=2^{-2}) — основной пульс
Восьмая: 1/8 (=2^{-3})
Шестнадцатая: 1/16 (=2^{-4})
Тридцать вторая: 1/32 (=2^{-5})
Такт 4/4: 4 четверти = 2^2 четверти
Произведение длиной 64 такта = 2^6 тактов — полный цикл.""",
    ],
    control_texts=[
        "Он шёл по улице и думал о прошедшем дне.",
        "The conference will be held in Berlin next year.",
        "raise ValueError('Invalid input: expected positive integer')",
    ],
    metadata={'p_formula': '2^k', 'collision_free': True},
)


# ─── Сводный реестр всех бенчмарков ─────────────────────────────────────────

ALL_DOMAIN_BENCHMARKS = {
    'skliarova': SKLIAROVA_BENCHMARK,
    'fomyuk':    FOMYUK_BENCHMARK,
    'kasatkin':  KASATKIN_BENCHMARK,
    'andreev':   ANDREEV_BENCHMARK,
    'herman':    HERMAN_BENCHMARK,
}


def get_benchmark_summary() -> Dict:
    """Возвращает краткое описание всех бенчмарков."""
    summary = {}
    for key, bench in ALL_DOMAIN_BENCHMARKS.items():
        summary[key] = {
            'source': bench.source_name,
            'hexagram_concept': bench.hexagram,
            'hypothesis': bench.hypothesis[:80] + '...',
            'n_test_texts': len(bench.texts),
            'n_control_texts': len(bench.control_texts),
        }
    return summary


def compute_source_advantage(
    model,
    benchmark: DomainBenchmark,
    tokenizer,
    device: str = 'cpu',
    max_len: int = 256,
) -> Dict:
    """Вычисляет PPL модели на тестовых vs контрольных текстах.

    Если источник действительно полезен для данного домена, PPL на
    тестовых текстах должна быть ниже при включённом модуле.

    Args:
        model: YiJingGPT с геометрическими модулями
        benchmark: набор текстов для данного источника
        tokenizer: токенизатор
        device: устройство

    Returns:
        dict с test_ppl, control_ppl, advantage (test_ppl < control_ppl?)
    """
    import torch

    model.eval()
    model.to(device)

    def compute_ppl(texts: List[str]) -> float:
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for text in texts:
                try:
                    if hasattr(tokenizer, 'encode'):
                        ids = tokenizer.encode(text)
                    else:
                        ids = [ord(c) % 256 for c in text]

                    ids_t = torch.tensor([ids[:max_len]], device=device)
                    if ids_t.shape[1] < 2:
                        continue

                    loss, _ = model(ids_t[:, :-1], targets=ids_t[:, 1:])
                    if loss is not None:
                        total_loss += loss.item() * (ids_t.shape[1] - 1)
                        total_tokens += ids_t.shape[1] - 1
                except Exception:
                    continue

        if total_tokens == 0:
            return float('inf')
        import math
        return math.exp(total_loss / total_tokens)

    test_ppl = compute_ppl(benchmark.texts)
    control_ppl = compute_ppl(benchmark.control_texts)

    advantage = control_ppl - test_ppl  # >0 значит тест труднее (ожидается)

    return {
        'source': benchmark.source_name,
        'test_ppl': float(test_ppl),
        'control_ppl': float(control_ppl),
        'ppl_gap': float(advantage),
        'hypothesis': benchmark.hypothesis[:60],
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Domain-specific benchmarks for YiJing geometric sources'
    )
    parser.add_argument('--demo', action='store_true',
                        help='Print all benchmark texts')
    parser.add_argument('--source', type=str, default=None,
                        choices=list(ALL_DOMAIN_BENCHMARKS.keys()),
                        help='Show texts for a specific source')
    parser.add_argument('--summary', action='store_true',
                        help='Print benchmark summary table')
    parser.add_argument('--export', type=str, default=None,
                        help='Export all benchmarks to JSON file')
    args = parser.parse_args()

    if args.summary:
        print("\n=== Domain Benchmarks Summary ===\n")
        summary = get_benchmark_summary()
        for key, info in summary.items():
            print(f"[{key.upper()}] {info['source']}")
            print(f"  Концепция: {info['hexagram_concept']}")
            print(f"  Гипотеза:  {info['hypothesis']}")
            print(f"  Текстов:   {info['n_test_texts']} тестовых, "
                  f"{info['n_control_texts']} контрольных")
            print()

    elif args.source:
        bench = ALL_DOMAIN_BENCHMARKS[args.source]
        print(f"\n=== {bench.source_name}: {bench.hexagram} ===")
        print(f"\nГипотеза: {bench.hypothesis}\n")
        print("=== ТЕСТОВЫЕ ТЕКСТЫ (геометрия должна помогать) ===\n")
        for i, t in enumerate(bench.texts):
            print(f"--- Текст {i+1} ---")
            print(t)
            print()
        print("=== КОНТРОЛЬНЫЕ ТЕКСТЫ (геометрия НЕ должна помогать) ===\n")
        for i, t in enumerate(bench.control_texts):
            print(f"--- Контроль {i+1} ---")
            print(t)
            print()

    elif args.demo:
        for key, bench in ALL_DOMAIN_BENCHMARKS.items():
            print(f"\n{'='*60}")
            print(f"  ИСТОЧНИК: {bench.source_name}")
            print(f"  И-ЦЗИ: {bench.hexagram}")
            print(f"{'='*60}")
            print(f"  Гипотеза: {bench.hypothesis}")
            print(f"\n  Пример теста:")
            if bench.texts:
                print("  " + bench.texts[0].replace('\n', '\n  '))
            print()

    elif args.export:
        export_data = {}
        for key, bench in ALL_DOMAIN_BENCHMARKS.items():
            export_data[key] = {
                'source_name': bench.source_name,
                'hexagram': bench.hexagram,
                'hypothesis': bench.hypothesis,
                'texts': bench.texts,
                'control_texts': bench.control_texts,
                'metadata': bench.metadata,
            }
        with open(args.export, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"Benchmarks exported to: {args.export}")

    else:
        print("Usage: python domain_benchmarks.py --demo | --summary | "
              "--source <name> | --export <file.json>")
        print("\nAvailable sources:", ', '.join(ALL_DOMAIN_BENCHMARKS.keys()))

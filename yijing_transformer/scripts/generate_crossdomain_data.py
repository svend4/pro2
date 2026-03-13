#!/usr/bin/env python3
"""
Generate cross-domain training data for NautilusMoME Phase 6.

Creates "poetic technical descriptions" and "technical poetic descriptions"
by combining aphorisms with code patterns, and surrealist imagery with algorithms.

The idea: surrealist poetry encodes structure beneath unexpected imagery.
Like biophysics uses physics formulas for biology, these cross-domain texts
teach the model to find analogies between domains.

Data types:
  1. PROVERB→CODE: Aphorism + its technical implementation
  2. CODE→PROVERB: Technical concept + its poetic metaphor
  3. SURREAL→TECH: Surrealist imagery as algorithm description
  4. POLARITY→PATTERN: Behavioral polarities as design patterns
"""

import os
import random
import glob


def load_aphorisms(base_dir='/tmp/info3/афоризмы'):
    """Load all aphorisms organized by category."""
    categories = {}
    for txt in glob.glob(os.path.join(base_dir, '*/классические-и-современные.txt')):
        cat = os.path.basename(os.path.dirname(txt))
        try:
            with open(txt, 'r', encoding='utf-8') as f:
                lines = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('##'):
                        # Extract quoted aphorisms
                        if '"' in line:
                            parts = line.split('"')
                            for i in range(1, len(parts), 2):
                                if len(parts[i]) > 10:
                                    lines.append(parts[i])
                        elif len(line) > 15 and not line.startswith('—'):
                            lines.append(line)
                if lines:
                    categories[cat] = lines
        except Exception:
            pass
    return categories


# Technical concepts mapped to poetic metaphors
TECH_POETRY_PAIRS = [
    # (technical description, poetic/surrealist equivalent)
    (
        "def sort(arr):\n    # O(n log n) — разделяй и властвуй",
        "# Сортировка — река, разделяющая камни по размеру\n"
        "# Быстрые потоки несут мелкие, тяжёлые оседают на дно\n"
        "# O(n log n) — как дерево: каждая ветвь делит мир пополам"
    ),
    (
        "class NeuralNetwork:\n    def forward(self, x):\n        # слой за слоем",
        "# Нейросеть — лес, где каждое дерево — нейрон\n"
        "# Корни пьют данные, листья выдыхают предсказания\n"
        "# Обратное распространение — осенний ветер, возвращающий листья к корням"
    ),
    (
        "# Рекурсия: функция вызывает саму себя\ndef factorial(n):\n    return n * factorial(n-1)",
        "# Рекурсия — зеркало напротив зеркала\n"
        "# Каждое отражение чуть меньше предыдущего\n"
        "# Пока не останется точка — базовый случай — зерно"
    ),
    (
        "# Хеш-таблица: O(1) поиск\nhash_map = {key: value}",
        "# Хеш-таблица — библиотека, где каждая книга\n"
        "# Знает своё место по имени, а не по номеру\n"
        "# Коллизия — два человека с одинаковыми именами в одной комнате"
    ),
    (
        "# Gradient descent: шаг за шагом к минимуму\nloss.backward()\noptimizer.step()",
        "# Градиентный спуск — слепой путник на горе\n"
        "# Каждый шаг — по склону, где ноги чувствуют уклон\n"
        "# Learning rate — длина шага: слишком большой — упадёшь в пропасть"
    ),
    (
        "# Attention mechanism\nattn = softmax(Q @ K.T / sqrt(d)) @ V",
        "# Внимание — фонарь в тёмной комнате слов\n"
        "# Query спрашивает: 'Кто здесь важен?'\n"
        "# Key отвечает: 'Я!', а Value — то, что он знает"
    ),
    (
        "# Binary search: отбрасывай половину\nmid = (lo + hi) // 2",
        "# Бинарный поиск — мудрец, который\n"
        "# Всегда спрашивает: 'Больше или меньше?'\n"
        "# И каждый ответ отсекает половину вселенной"
    ),
    (
        "# Deadlock: два процесса ждут друг друга\nlock_a.acquire(); lock_b.acquire()",
        "# Дедлок — два рыцаря в узком коридоре\n"
        "# Каждый ждёт, что другой отступит первым\n"
        "# И оба стоят вечно, как две скалы"
    ),
    (
        "# Transformer: self-attention + FFN\nclass TransformerBlock:\n    def forward(self, x):",
        "# Трансформер — собрание мудрецов за круглым столом\n"
        "# Каждый слушает всех, но больше — тех, кто говорит о главном\n"
        "# FFN — время размышления в одиночестве после дискуссии"
    ),
    (
        "# MoE: Mixture of Experts\nrouter(x) → expert_i(x)",
        "# Смесь экспертов — совет врачей у постели больного\n"
        "# Терапевт видит одно, хирург — другое, психолог — третье\n"
        "# Маршрутизатор — главврач, который решает, кого позвать"
    ),
]

# Surrealist → Technical mappings (inspired by Stefan Engel)
SURREAL_TECH_MAPS = [
    (
        "следы скрепок прерывают траекторию чернил",
        "# Прерывание: interrupt handler перехватывает поток выполнения\n"
        "# Скрепки = hardware interrupts, чернила = instruction pipeline\n"
        "# Траектория прервана, но продолжится после обработки"
    ),
    (
        "слёз потоки замерзают превращаясь в узелковое письмо",
        "# Сериализация: поток данных (слёзы) замораживается (freeze)\n"
        "# и превращается в компактное представление (узелковое письмо)\n"
        "# Десериализация — оттаивание: bytes → objects"
    ),
    (
        "у балерин на шахматной доске блестят ресницы искрами комет",
        "# Многопоточность: балерины = threads на шахматной доске = scheduler\n"
        "# Искры комет = context switches, блестящие но затратные\n"
        "# Каждая балерина танцует свой танец, но на общей доске"
    ),
    (
        "луна компилирует тишину",
        "# Компиляция: исходный код (тишина) → машинный код (лунный свет)\n"
        "# Компилятор (луна) преобразует абстракцию в конкретику\n"
        "# Тишина = source, лунный свет = binary"
    ),
    (
        "деревья парсят ветер",
        "# Парсинг: дерево разбора (AST) анализирует поток токенов (ветер)\n"
        "# Каждая ветвь = правило грамматики\n"
        "# Листья = терминалы, ствол = корневой узел"
    ),
    (
        "река выполняет рекурсию",
        "# Рекурсия потоков: река разветвляется (fork)\n"
        "# Каждый приток вызывает себя с уменьшенным бассейном\n"
        "# Устье = base case, дельта = return value"
    ),
    (
        "горстка снега сжимает в объятиях заплатки ступеней",
        "# Сжатие данных: горстка (compressed) содержит всю информацию\n"
        "# Ступени = уровни квантования\n"
        "# Объятия = lossy compression, заплатки = residuals"
    ),
    (
        "птичьи перья на деревьях шепчут что-то облакам",
        "# Микросервисы: перья (lightweight services) на деревьях (hosts)\n"
        "# Шепчут облакам = отправляют данные в cloud\n"
        "# Каждое перо легко заменить, дерево устоит"
    ),
]

# MBTI → Design Pattern mappings
MBTI_PATTERNS = [
    ("INTJ (Стратег)", "Singleton",
     "# INTJ = Singleton: один экземпляр, всё контролирует\n"
     "# Стратег планирует систему, не нуждаясь в копиях\n"
     "# private constructor — как закрытая дверь кабинета мыслителя"),
    ("ENFP (Борец)", "Observer",
     "# ENFP = Observer: слушает все события, реагирует на каждое\n"
     "# Энтузиаст подписывается на всё, что интересно\n"
     "# event.subscribe() — как рука, поднятая в ответ на каждый вопрос"),
    ("ISTP (Виртуоз)", "Factory",
     "# ISTP = Factory: создаёт объекты руками, разбирает и собирает\n"
     "# Мастер не теоретизирует — он делает\n"
     "# create_product(type) — как инструмент, подобранный под задачу"),
    ("ENFJ (Наставник)", "Mediator",
     "# ENFJ = Mediator: координирует взаимодействие между объектами\n"
     "# Лидер, который не командует, а соединяет\n"
     "# mediator.notify() — как тёплое слово, связывающее людей"),
    ("INTP (Учёный)", "Strategy",
     "# INTP = Strategy: алгоритм можно заменить на лету\n"
     "# Аналитик пробует разные подходы к одной задаче\n"
     "# set_strategy(new_algo) — как смена гипотезы в эксперименте"),
    ("ESFJ (Консул)", "Adapter",
     "# ESFJ = Adapter: соединяет несовместимые интерфейсы\n"
     "# Дипломат делает так, чтобы все могли работать вместе\n"
     "# adapt(old_interface) → new_interface — как перевод между языками"),
]


def generate_crossdomain_texts(aphorisms, n_samples=500):
    """Generate cross-domain training texts."""
    texts = []

    # 1. PROVERB → CODE: aphorism + technical parallel
    tech_categories = [
        'искусственный-интеллект', 'нейронные-сети', 'алгоритмы',
        'программирование', 'квантовые-вычисления', 'робототехника',
        'облачные-технологии', 'кибербезопасность', 'блокчейн-и-криптовалюты',
        'компьютерное-зрение', 'обработка-естественного-языка',
    ]
    human_categories = [
        'мудрость-и-глупость', 'сны-и-сновидения', 'интуиция-и-разум',
        'терпение-и-настойчивость', 'простота-и-сложность',
        'порядок-и-хаос', 'память-и-забвение', 'искусство-и-творчество',
    ]

    for _ in range(n_samples // 5):
        # Pick a tech aphorism and a human one, combine
        tech_cat = random.choice([c for c in tech_categories if c in aphorisms])
        human_cat = random.choice([c for c in human_categories if c in aphorisms])
        if tech_cat and human_cat:
            tech_aph = random.choice(aphorisms[tech_cat])
            human_aph = random.choice(aphorisms[human_cat])
            text = (f"# Кросс-доменная аналогия\n"
                    f"# Техническое: {tech_aph}\n"
                    f"# Гуманитарное: {human_aph}\n"
                    f"# Мост: оба говорят об одном — о поиске закономерностей\n")
            texts.append(text)

    # 2. CODE → POETRY pairs
    for tech, poetry in TECH_POETRY_PAIRS:
        # Forward: tech → poetry
        texts.append(f"# === Технический процесс ===\n{tech}\n\n"
                     f"# === Поэтическое описание ===\n{poetry}\n")
        # Reverse: poetry → tech
        texts.append(f"# === Поэтический образ ===\n{poetry}\n\n"
                     f"# === Техническая реализация ===\n{tech}\n")

    # 3. SURREAL → TECH mappings
    for surreal, tech in SURREAL_TECH_MAPS:
        texts.append(f"# === Сюрреалистический образ ===\n# \"{surreal}\"\n\n"
                     f"# === Техническая интерпретация ===\n{tech}\n")

    # 4. MBTI → Design Pattern
    for mbti, pattern, desc in MBTI_PATTERNS:
        texts.append(f"# === Психотип → Паттерн проектирования ===\n"
                     f"# {mbti} = {pattern}\n{desc}\n")

    # 5. Mixed aphorism chains
    all_cats = list(aphorisms.keys())
    for _ in range(n_samples // 5):
        cats = random.sample(all_cats, min(3, len(all_cats)))
        chain = "# === Цепочка аналогий ===\n"
        for cat in cats:
            aph = random.choice(aphorisms[cat])
            chain += f"# [{cat}]: {aph}\n"
        chain += "# Вывод: все формулы описывают одну реальность разными словами\n"
        texts.append(chain)

    random.shuffle(texts)
    return texts


def main():
    print("Loading aphorisms...")
    aphorisms = load_aphorisms()
    print(f"  Loaded {len(aphorisms)} categories, "
          f"{sum(len(v) for v in aphorisms.values())} aphorisms total")

    # Show sample categories
    for cat in sorted(aphorisms.keys())[:10]:
        print(f"  {cat}: {len(aphorisms[cat])} aphorisms")
    print(f"  ... and {len(aphorisms) - 10} more categories")
    print()

    print("Generating cross-domain training data...")
    texts = generate_crossdomain_texts(aphorisms, n_samples=500)
    print(f"  Generated {len(texts)} cross-domain texts")

    # Save
    output_dir = '/tmp/crossdomain_data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'crossdomain_training.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n\n')

    total_chars = sum(len(t) for t in texts)
    print(f"  Saved to {output_path}")
    print(f"  Total: {total_chars:,} chars, ~{total_chars//4:,} tokens (est.)")

    # Show samples
    print("\n=== SAMPLE OUTPUTS ===\n")
    for i, text in enumerate(texts[:5]):
        print(f"--- Sample {i+1} ---")
        print(text[:300])
        print()


if __name__ == '__main__':
    main()

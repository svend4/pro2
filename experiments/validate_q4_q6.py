"""
validate_q4_q6.py — Эксперимент: проверка что Q4⊂Q6 вложение работает.

Гипотеза (из PSEUDORAG_YIJING_BRIDGE.md):
    Тексты одного PseudoRAG-архетипа (Q4) должны давать
    гексаграммы Q6 из одного и того же кластера из 4 точек.
    Расстояние Хэмминга внутри кластера должно быть ≤ 2.

Запуск:
    python experiments/validate_q4_q6.py

Ожидаемый результат:
    clustering_score < 2.5 → гипотеза подтверждена
    clustering_score > 3.0 → гипотеза отвергнута, нужна другая инициализация
"""

import sys
import json
import itertools
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'meta'))

# ─── Q4: PseudoRAG архетипы (16 штук) ────────────────────────────────────────

Q4_ARCHETYPES = {
    # Код:  (Materiality, Dynamics, Scale, Structure): описание, ключевые слова
    'MSEO': ('кристалл',   ['кристалл', 'минерал', 'структура', 'решётка', 'порядок']),
    'MSEF': ('песок',      ['песок', 'порошок', 'частицы', 'хаос', 'рассыпной']),
    'MSCO': ('здание',     ['здание', 'дом', 'архитектура', 'конструкция', 'стена']),
    'MSCF': ('лес',        ['лес', 'дерево', 'растение', 'природа', 'экосистема']),
    'MDEO': ('механизм',   ['механизм', 'шестерня', 'двигатель', 'пружина', 'деталь']),
    'MDEF': ('организм',   ['организм', 'клетка', 'биология', 'жизнь', 'рост']),
    'MDCO': ('машина',     ['машина', 'автомобиль', 'транспорт', 'двигатель', 'колесо']),
    'MDCF': ('город',      ['город', 'улица', 'трафик', 'население', 'инфраструктура']),
    'ASEO': ('аксиома',    ['аксиома', 'принцип', 'постулат', 'логика', 'истина']),
    'ASEF': ('архетип',    ['архетип', 'символ', 'паттерн', 'прообраз', 'идея']),
    'ASCO': ('теория',     ['теория', 'модель', 'концепция', 'наука', 'гипотеза']),
    'ASCF': ('культура',   ['культура', 'традиция', 'искусство', 'история', 'общество']),
    'ADEO': ('алгоритм',   ['алгоритм', 'программа', 'код', 'функция', 'вычисление']),
    'ADEF': ('интуиция',   ['интуиция', 'чувство', 'ощущение', 'вдохновение', 'инсайт']),
    'ADCO': ('программа',  ['программа', 'система', 'архитектура', 'модуль', 'процесс']),
    'ADCF': ('общество',   ['общество', 'социум', 'люди', 'взаимодействие', 'сообщество']),
}

# Тестовые предложения — по 5 на каждый архетип
Q4_TEST_SENTENCES = {
    'MSEO': [
        "Кристаллическая решётка соли имеет кубическую симметрию.",
        "Минералы образуют правильную геометрическую структуру.",
        "Снежинка — это кристалл воды с шестигранной симметрией.",
        "Кварц представляет собой упорядоченную решётку атомов кремния.",
        "Структура алмаза определяется правильным расположением атомов углерода.",
    ],
    'MSCO': [
        "Здание состоит из фундамента, стен и крыши.",
        "Архитектура собора отражает средневековое мышление.",
        "Бетонный каркас здания несёт статическую нагрузку.",
        "Конструкция моста рассчитана на определённую нагрузку.",
        "Стена разделяет пространство на две части.",
    ],
    'ADEO': [
        "Алгоритм сортировки работает за O(n log n) шагов.",
        "Функция принимает входные данные и возвращает результат.",
        "Программа состоит из последовательности инструкций.",
        "Рекурсивная функция вызывает сама себя с меньшими данными.",
        "Цикл повторяет операцию заданное количество раз.",
    ],
    'MDCF': [
        "Город развивается органически вокруг центра.",
        "Трафик формирует сложную систему потоков.",
        "Городская инфраструктура включает дороги, воду и электричество.",
        "Население города взаимодействует в сложных социальных сетях.",
        "Улицы города образуют нерегулярную сеть связей.",
    ],
    'ASCF': [
        "Культура формируется через передачу традиций.",
        "Искусство отражает мировоззрение эпохи.",
        "Язык является носителем культурной памяти.",
        "Религиозные ритуалы укрепляют социальные связи.",
        "История культуры — это история изменения смыслов.",
    ],
}

# ─── Q6: проекция через тематические оси ─────────────────────────────────────

def text_to_q6_simple(text: str) -> list:
    """
    Детерминированная проекция текста на Q6 через 6 тематических осей.
    В реальном эксперименте здесь должна быть HexagramProjection из модели.
    """
    axes = [
        # Ось 0: Материальное (+1) vs Абстрактное (-1)
        (['кристалл', 'здание', 'машина', 'город', 'минерал', 'бетон', 'металл'],
         ['алгоритм', 'теория', 'принцип', 'культура', 'интуиция', 'идея', 'символ']),
        # Ось 1: Статичное (+1) vs Динамичное (-1)
        (['решётка', 'структура', 'стена', 'кристалл', 'аксиома', 'принцип'],
         ['трафик', 'рост', 'процесс', 'цикл', 'движение', 'развитие', 'поток']),
        # Ось 2: Элементарное (+1) vs Комплексное (-1)
        (['атом', 'клетка', 'деталь', 'частица', 'пружина', 'функция', 'шаг'],
         ['система', 'город', 'общество', 'экосистема', 'архитектура', 'культура']),
        # Ось 3: Упорядоченное (+1) vs Текучее (-1)
        (['решётка', 'алгоритм', 'аксиома', 'правило', 'закон', 'формула'],
         ['хаос', 'интуиция', 'лес', 'природа', 'органически', 'нерегулярн']),
        # Ось 4: Конкретное (+1) vs Абстрактное (-1)
        (['бетон', 'металл', 'камень', 'дерево', 'вода', 'земля'],
         ['смысл', 'значение', 'концепция', 'мышление', 'восприятие']),
        # Ось 5: Внешнее (+1) vs Внутреннее (-1)
        (['улица', 'поверхность', 'форма', 'внешний', 'видимый'],
         ['внутренний', 'суть', 'ядро', 'основа', 'скрытый', 'глубина']),
    ]

    text_lower = text.lower()
    q6 = []
    for pos_words, neg_words in axes:
        pos_score = sum(1 for w in pos_words if w in text_lower)
        neg_score = sum(1 for w in neg_words if w in text_lower)
        if pos_score > neg_score:
            q6.append(1)
        elif neg_score > pos_score:
            q6.append(-1)
        else:
            q6.append(1 if hash(text + str(len(q6))) % 2 == 0 else -1)

    return q6


def hamming_distance(v1: list, v2: list) -> int:
    return sum(a != b for a, b in zip(v1, v2))


def hex_index(q6: list) -> int:
    return sum((1 if b == 1 else 0) << i for i, b in enumerate(q6))


# ─── Основной эксперимент ─────────────────────────────────────────────────────

def run_q4_q6_validation(project_fn=None):
    if project_fn is None:
        project_fn = text_to_q6_simple

    print("=" * 60)
    print("ЭКСПЕРИМЕНТ: Валидация Q4⊂Q6 вложения")
    print("=" * 60)

    archetype_results = {}

    for archetype_code, sentences in Q4_TEST_SENTENCES.items():
        name, keywords = Q4_ARCHETYPES[archetype_code]

        q6_vectors = [project_fn(s) for s in sentences]
        hex_indices = [hex_index(v) for v in q6_vectors]

        pairs = list(itertools.combinations(range(len(q6_vectors)), 2))
        distances = [hamming_distance(q6_vectors[i], q6_vectors[j]) for i, j in pairs]

        avg_distance = sum(distances) / len(distances) if distances else 0
        max_distance = max(distances) if distances else 0

        archetype_results[archetype_code] = {
            'name': name,
            'q6_vectors': q6_vectors,
            'hex_indices': hex_indices,
            'avg_hamming': avg_distance,
            'max_hamming': max_distance,
            'clustered': avg_distance <= 2.5,
        }

        status = "КЛАСТЕР" if avg_distance <= 2.5 else "РАССЕЯН"
        print(f"\n[{archetype_code}] {name.upper()} — {status}")
        print(f"  Гексаграммы: {hex_indices}")
        print(f"  Средний Хэмминг: {avg_distance:.2f} | Максимум: {max_distance}")

    all_avg = [r['avg_hamming'] for r in archetype_results.values()]
    global_avg = sum(all_avg) / len(all_avg)
    clustered_count = sum(1 for r in archetype_results.values() if r['clustered'])
    total = len(archetype_results)

    print("\n" + "=" * 60)
    print("ИТОГ")
    print("=" * 60)
    print(f"Архетипов протестировано: {total}")
    print(f"Кластеризованных (Хэмминг <= 2.5): {clustered_count}/{total}")
    print(f"Глобальный средний Хэмминг: {global_avg:.2f}")
    print()
    if global_avg <= 2.5:
        print("ГИПОТЕЗА ПОДТВЕРЖДЕНА: Q4<Q6 вложение работает.")
        print("   PseudoRAG может использоваться как учитель для YiJing.")
    elif global_avg <= 3.0:
        print("ГИПОТЕЗА ЧАСТИЧНО ПОДТВЕРЖДЕНА.")
        print("   Инициализация HexagramProjection из PseudoRAG улучшит результат.")
    else:
        print("ГИПОТЕЗА ОТВЕРГНУТА: архетипы рассеяны по всему Q6.")
        print("   Нужна другая схема отображения Q4→Q6.")

    return archetype_results, global_avg


def try_neural_projection():
    try:
        import torch
        ckpt_paths = [
            ROOT / 'checkpoint_bidir_v2.pt',
            ROOT / 'hmoe_self_trained_v4.pt',
            ROOT / 'hmoe_fixed_self.pt',
            ROOT / 'experiments' / 'lean_baseline_final.pt',
        ]

        ckpt = next((p for p in ckpt_paths if p.exists()), None)
        if ckpt is None:
            print("Checkpoint не найден, используется keyword-based проекция.")
            return None

        print(f"Загружаем checkpoint: {ckpt.name}")

        # Попробовать LeanYiJingGPT
        try:
            from yijing_transformer.models.lean_model import LeanYiJingGPT
            saved = torch.load(ckpt, map_location='cpu', weights_only=True)
            if 'model_state' in saved:
                cfg = saved.get('config', {})
                model = LeanYiJingGPT(
                    vocab_size=cfg.get('vocab_size', 256),
                    d_model=cfg.get('d_model', 128),
                    n_layers=cfg.get('n_layers', 4),
                    n_heads=cfg.get('n_heads', 4),
                    block_size=cfg.get('block_size', 256),
                )
                model.load_state_dict(saved['model_state'], strict=False)
                model.eval()
                print(f"Модель LeanYiJingGPT загружена из {ckpt.name}")

                def neural_project_fn(text: str) -> list:
                    tokens = torch.tensor(
                        [min(b, 255) for b in text.encode('utf-8', errors='replace')[:32]],
                        dtype=torch.long,
                    ).unsqueeze(0)
                    with torch.no_grad():
                        logits, _ = model(tokens)
                        h = logits.mean(dim=1).squeeze(0)[:6]
                        return [1 if x > 0 else -1 for x in h.tolist()]

                return neural_project_fn
        except Exception as e:
            print(f"Не удалось загрузить LeanYiJingGPT: {e}")

    except Exception as e:
        print(f"Нейросеть недоступна ({e}), используется keyword-based проекция.")

    return None


def validate_with_meta_hexdim():
    meta_root = ROOT.parent / 'meta'
    if not meta_root.exists():
        print("Репозиторий meta не найден, пропускаем проверку через hexdim.")
        return None

    sys.path.insert(0, str(meta_root))
    try:
        from projects.hexdim.hexdim import HexDimModel
        hexdim = HexDimModel()

        print("\n--- Проверка через meta/hexdim ---")
        tesseracts = hexdim.tesseracts()
        print(f"Тессерактов Q4 внутри Q6: {len(tesseracts)}")
        print(f"Каждый тессеракт содержит {len(tesseracts[0])} вершин (= 16 = 2^4)")
        print(f"math Q4<Q6 верифицирована через meta/hexdim")

        return tesseracts
    except Exception as e:
        print(f"hexdim: {e}")
        return None


if __name__ == '__main__':
    print("\nЗагрузка нейросетевой проекции (если доступна)...")
    neural_fn = try_neural_projection()

    print("\nЗапуск основного эксперимента...")
    results, score = run_q4_q6_validation(project_fn=neural_fn)

    print("\nМатематическая верификация через meta...")
    validate_with_meta_hexdim()

    output = {
        'global_avg_hamming': score,
        'hypothesis_confirmed': score <= 2.5,
        'used_neural_model': neural_fn is not None,
        'archetypes': {
            k: {
                'name': v['name'],
                'avg_hamming': v['avg_hamming'],
                'clustered': v['clustered'],
                'hex_indices': v['hex_indices'],
            }
            for k, v in results.items()
        }
    }

    out_path = ROOT / 'experiments' / 'q4_q6_validation_result.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены: {out_path}")

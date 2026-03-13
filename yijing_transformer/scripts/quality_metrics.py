"""
quality_metrics.py — метрики качества за пределами PPL.

Восстанавливает измерения, потерянные в v59–v61:
  1. source_diversity     — насколько разные сигналы дают разные модули
  2. trit_differentiation — насколько по-разному источники голосуют за архетипы
  3. geometric_coherence  — соответствие attention и Хэмминг-метрики Q6
  4. twilight_language_rate — частота «сумеречных» неологизмов в генерации
  5. hexagram_expert_alignment — корреляция выхода эксперта с его Q6-якорем

Использование:
    python scripts/quality_metrics.py --model checkpoint.pt --text sample.txt

Или как библиотека:
    from scripts.quality_metrics import compute_all_metrics
    report = compute_all_metrics(model, tokenizer, sample_texts)
"""

import os
import sys
import math
import json
import argparse
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F


# ─── 1. Source Diversity ──────────────────────────────────────────────────────

def compute_source_diversity(source_outputs: List[torch.Tensor]) -> Dict:
    """Измеряет насколько по-разному N источников обрабатывают одни и те же токены.

    Высокое разнообразие = источники видят вход по-разному = потенциально полезны.
    Низкое разнообразие = источники дублируют друг друга = вредные источники.

    Args:
        source_outputs: list of N tensors (B, T, d_model)

    Returns:
        dict с ключами:
            mean_pairwise_cosine_sim: средняя косинусная схожесть между парами
            mean_l2_std: средний std по оси источников (в L2-метрике)
            max_outlier_distance: расстояние самого «непохожего» источника
            diversity_score: агрегированная оценка [0..1], выше = разнообразнее
    """
    if not source_outputs:
        return {'diversity_score': 0.0}

    N = len(source_outputs)
    stacked = torch.stack(source_outputs, dim=0)  # (N, B, T, d_model)

    # L2 std по оси источников: для каждого (batch, token, dim) смотрим разброс
    l2_std = stacked.std(dim=0).mean().item()  # скаляр

    # Среднеагрегированный вектор каждого источника
    mean_per_source = stacked.mean(dim=(1, 2))  # (N, d_model)
    mean_per_source_norm = F.normalize(mean_per_source, dim=-1)  # (N, d_model)

    # Попарная косинусная схожесть
    sim_matrix = mean_per_source_norm @ mean_per_source_norm.T  # (N, N)
    # Исключаем диагональ (самосравнение = 1.0)
    mask = ~torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
    if mask.any():
        mean_pairwise_sim = sim_matrix[mask].mean().item()
    else:
        mean_pairwise_sim = 1.0

    # Расстояние до центроида
    centroid = mean_per_source_norm.mean(dim=0, keepdim=True)  # (1, d_model)
    distances = (mean_per_source_norm - centroid).norm(dim=-1)  # (N,)
    max_outlier = distances.max().item()

    # diversity_score: инвертированная схожесть, нормированная на [0,1]
    # 1.0 = все источники максимально разные, 0.0 = все одинаковые
    diversity_score = float(1.0 - (mean_pairwise_sim + 1.0) / 2.0)

    return {
        'mean_pairwise_cosine_sim': float(mean_pairwise_sim),
        'mean_l2_std': float(l2_std),
        'max_outlier_distance': float(max_outlier),
        'diversity_score': float(diversity_score),
        'n_sources': N,
    }


# ─── 2. Trit Differentiation ─────────────────────────────────────────────────

def compute_trit_differentiation(trit_scores_per_source: List[torch.Tensor]) -> Dict:
    """Измеряет дифференциацию тернарного голосования между источниками.

    Если все источники голосуют одинаково (все +1, или все 0, или все -1),
    это указывает на общий trit_proj (старый баг). После исправления
    разные источники должны давать разные паттерны голосования.

    Args:
        trit_scores_per_source: list of N tensors (B, n_archetypes)

    Returns:
        dict с ключами:
            mean_inter_source_std: std тритов по оси источников (выше = лучше)
            trit_agreement_rate: доля архетипов, где все источники согласны
            differentiation_score: агрегированная оценка [0..1]
    """
    if len(trit_scores_per_source) < 2:
        return {'differentiation_score': 0.0}

    stacked = torch.stack(trit_scores_per_source, dim=0)  # (N, B, n_archetypes)
    hard = stacked.sign()  # приводим к {-1, 0, +1}

    # Std по оси источников: насколько разные источники голосуют по-разному
    inter_source_std = stacked.std(dim=0).mean().item()

    # Доля архетипов, где все источники одновременно согласны (все имеют одинаковый знак)
    N = stacked.shape[0]
    # Считаем число уникальных значений для каждого (batch, archetype)
    agreement_count = 0
    total_count = 0
    for b in range(hard.shape[1]):
        for a in range(hard.shape[2]):
            col = hard[:, b, a]
            unique_vals = col.unique()
            if len(unique_vals) == 1:
                agreement_count += 1
            total_count += 1

    agreement_rate = float(agreement_count) / max(total_count, 1)

    # differentiation_score: высокий std + низкое agreement = хорошо
    differentiation_score = float(inter_source_std) * (1.0 - agreement_rate)
    # Нормируем на ожидаемый диапазон [0..1]
    differentiation_score = min(differentiation_score, 1.0)

    return {
        'mean_inter_source_std': float(inter_source_std),
        'trit_agreement_rate': float(agreement_rate),
        'differentiation_score': float(differentiation_score),
        'n_sources': len(trit_scores_per_source),
    }


# ─── 3. Geometric Coherence ───────────────────────────────────────────────────

def compute_geometric_coherence(
    attention_patterns: torch.Tensor,
    codebook: torch.Tensor
) -> Dict:
    """Измеряет соответствие attention-паттернов Хэмминг-метрике Q6.

    Гипотеза: если геометрическая архитектура работает, токены, близкие в Q6
    (малое Хэмминг-расстояние), должны иметь высокий attention между собой.

    Args:
        attention_patterns: (B, H, T, T) или (B, T, T) — attention весы
        codebook: (vocab_size, 6) — Q6-векторы для каждого токена

    Returns:
        dict с ключами:
            pearson_r: корреляция Пирсона между -Хэмминг и log(attention)
            spearman_rho: ранговая корреляция
            coherence_score: агрегированная оценка [-1..1]
    """
    if attention_patterns.dim() == 4:
        # Усредняем по batch и heads
        attn = attention_patterns.mean(dim=(0, 1))  # (T, T)
    else:
        attn = attention_patterns.mean(dim=0)  # (T, T)

    T = attn.shape[0]
    if T > codebook.shape[0]:
        T = codebook.shape[0]
        attn = attn[:T, :T]

    # Хэмминг-расстояния между T позициями
    cb = codebook[:T].float()  # (T, 6), значения {-1, +1}
    # Хэмминг в Q6: d(a,b) = sum(a_i != b_i) / 6
    hamming = ((cb.unsqueeze(0) != cb.unsqueeze(1)).float().sum(dim=-1) / 6.0)  # (T, T)

    # Маска диагонали
    mask = ~torch.eye(T, dtype=torch.bool)
    neg_hamming = -hamming[mask].cpu().float()

    log_attn = (attn[mask] + 1e-10).log().cpu().float()

    if neg_hamming.std() < 1e-8 or log_attn.std() < 1e-8:
        return {'coherence_score': 0.0, 'note': 'degenerate (zero variance)'}

    # Pearson r
    def pearson(x, y):
        x = x - x.mean()
        y = y - y.mean()
        denom = x.norm() * y.norm()
        return float((x * y).sum() / (denom + 1e-10))

    r = pearson(neg_hamming, log_attn)

    # Spearman rho (через rankordering)
    def spearman(x, y):
        n = x.shape[0]
        rx = x.argsort().argsort().float()
        ry = y.argsort().argsort().float()
        return pearson(rx, ry)

    rho = spearman(neg_hamming, log_attn)

    return {
        'pearson_r': float(r),
        'spearman_rho': float(rho),
        'coherence_score': float((r + rho) / 2.0),
    }


# ─── 4. Twilight Language Rate ────────────────────────────────────────────────

def compute_twilight_language_rate(
    generated_texts: List[str],
    known_vocab: Optional[set] = None,
    min_word_len: int = 4,
) -> Dict:
    """Измеряет частоту появления «сумеречных» конструкций в генерации.

    «Сумеречный язык» — портманто, неологизмы, кросс-доменные слияния:
    «началость», «единологится», «чистость — мудрость».

    Признаки сумеречного токена:
    - Слово есть в генерации, но отсутствует в known_vocab
    - Слово содержит смешение корней из разных языков
    - Необычные словообразовательные суффиксы (-ость, -ность + глагольные корни)

    Args:
        generated_texts: список сгенерированных текстов
        known_vocab: известные слова (если None — определяется автоматически)
        min_word_len: минимальная длина слова для анализа

    Returns:
        dict с метриками сумеречного языка
    """
    import re

    # Русские суффиксы, характерные для «сумеречного» словообразования
    twilight_suffixes_ru = [
        'ость', 'ность', 'логится', 'ируется', 'логия', 'зация',
        'ённость', 'инность', 'чность',
    ]
    # Признаки кросс-доменных слияний (латиница + кириллица)
    mixed_script_pattern = re.compile(r'[a-zA-Z]+[а-яёА-ЯЁ]+|[а-яёА-ЯЁ]+[a-zA-Z]+')

    # Стандартные словари (русский + английский общеупотребительные)
    common_ru_endings = {'ие', 'ия', 'ть', 'ет', 'ит', 'ать', 'ые', 'ого'}

    all_words = []
    for text in generated_texts:
        words = re.findall(r'[а-яёА-ЯЁa-zA-Z]{' + str(min_word_len) + r',}', text)
        all_words.extend(words)

    if not all_words:
        return {'twilight_rate': 0.0, 'total_words': 0}

    twilight_count = 0
    novel_count = 0
    mixed_count = 0
    twilight_examples = []

    for word in all_words:
        is_twilight = False
        word_lower = word.lower()

        # Признак 1: необычные суффиксы в русских словах (неологизмы)
        for suffix in twilight_suffixes_ru:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 3:
                # Проверяем, что это не обычное слово
                base = word_lower[:-len(suffix)]
                if len(base) >= 3 and not any(base.endswith(e) for e in common_ru_endings):
                    is_twilight = True
                    break

        # Признак 2: смешение скриптов
        if mixed_script_pattern.match(word_lower):
            is_twilight = True
            mixed_count += 1

        # Признак 3: известный вокабуляр vs. нет
        if known_vocab and word_lower not in known_vocab and len(word_lower) >= min_word_len:
            novel_count += 1

        if is_twilight:
            twilight_count += 1
            if len(twilight_examples) < 20:
                twilight_examples.append(word)

    total = len(all_words)
    twilight_rate = float(twilight_count) / max(total, 1)
    novel_rate = float(novel_count) / max(total, 1) if known_vocab else None

    return {
        'twilight_rate': twilight_rate,
        'twilight_count': twilight_count,
        'mixed_script_count': mixed_count,
        'novel_word_count': novel_count,
        'novel_rate': novel_rate,
        'total_words': total,
        'twilight_examples': twilight_examples[:10],
        # Качественная оценка: > 5% = богатый сумеречный язык
        'richness': 'rich' if twilight_rate > 0.05
                    else 'moderate' if twilight_rate > 0.01
                    else 'sparse',
    }


# ─── 5. Hexagram–Expert Alignment ────────────────────────────────────────────

def compute_hexagram_expert_alignment(
    expert_outputs: Dict[str, torch.Tensor],
    q6_anchors: torch.Tensor,
    expert_names: List[str],
) -> Dict:
    """Измеряет совпадение выхода эксперта с его гексаграммным Q6-якорем.

    Если эксперт специализировался правильно, среднеагрегированный вектор
    его выходов должен коррелировать с его Q6-якорем (6-битным вектором).

    Args:
        expert_outputs: словарь {name: (B, T, d_model)}
        q6_anchors: (n_experts, 6) — Q6-якоря экспертов из HEXAGRAM_MAP
        expert_names: список имён в порядке, соответствующем q6_anchors

    Returns:
        dict per-expert корреляций и общий alignment_score
    """
    results = {}
    alignment_scores = []

    for i, name in enumerate(expert_names):
        if name not in expert_outputs:
            continue
        if i >= q6_anchors.shape[0]:
            continue

        out = expert_outputs[name]  # (B, T, d_model)
        anchor = q6_anchors[i]  # (6,)

        # Среднеагрегированный выход эксперта: (d_model,)
        mean_out = out.mean(dim=(0, 1))  # (d_model,)

        # Проецируем в Q6-пространство: берём первые 6 компонент
        # (это приближение — в полной версии нужен обученный Q6-проектор)
        if mean_out.shape[0] >= 6:
            projected = mean_out[:6]
            projected_norm = F.normalize(projected.unsqueeze(0), dim=-1).squeeze(0)
            anchor_norm = F.normalize(anchor.float().unsqueeze(0), dim=-1).squeeze(0)
            alignment = float((projected_norm * anchor_norm).sum())
        else:
            alignment = 0.0

        results[name] = {
            'q6_alignment': alignment,
            'hexagram': _get_hexagram_label(name),
        }
        alignment_scores.append(alignment)

    overall = float(sum(alignment_scores) / max(len(alignment_scores), 1))

    return {
        'per_expert': results,
        'mean_alignment': overall,
        # > 0.3 = хорошая специализация, > 0.6 = отличная
        'alignment_quality': 'strong' if overall > 0.6
                              else 'moderate' if overall > 0.3
                              else 'weak',
    }


def _get_hexagram_label(name: str) -> str:
    """Возвращает символ гексаграммы для имени эксперта."""
    labels = {
        'MATH': '乾(1)', 'CODE': '巽(57)', 'HUMAN': '坤(2)',
        'SYSTEM': '坎(29)', 'RECON': '離(30)', 'INFO': '兌(58)',
        'SYNTH': '革(49)',
    }
    return labels.get(name, '?')


# ─── 6. Aggregated Report ─────────────────────────────────────────────────────

def compute_all_metrics(
    source_outputs: Optional[List[torch.Tensor]] = None,
    trit_scores: Optional[List[torch.Tensor]] = None,
    attention_patterns: Optional[torch.Tensor] = None,
    codebook: Optional[torch.Tensor] = None,
    generated_texts: Optional[List[str]] = None,
    expert_outputs: Optional[Dict[str, torch.Tensor]] = None,
    q6_anchors: Optional[torch.Tensor] = None,
    expert_names: Optional[List[str]] = None,
) -> Dict:
    """Вычисляет все доступные метрики и возвращает единый отчёт.

    Каждый раздел опционален — передайте только доступные данные.

    Returns:
        Словарь со всеми доступными метриками и итоговым summary.
    """
    report = {}

    if source_outputs is not None:
        report['source_diversity'] = compute_source_diversity(source_outputs)

    if trit_scores is not None:
        report['trit_differentiation'] = compute_trit_differentiation(trit_scores)

    if attention_patterns is not None and codebook is not None:
        report['geometric_coherence'] = compute_geometric_coherence(
            attention_patterns, codebook
        )

    if generated_texts is not None:
        report['twilight_language'] = compute_twilight_language_rate(generated_texts)

    if expert_outputs is not None and q6_anchors is not None and expert_names is not None:
        report['hexagram_alignment'] = compute_hexagram_expert_alignment(
            expert_outputs, q6_anchors, expert_names
        )

    # Summary: общая оценка «линии передачи»
    scores = []
    if 'source_diversity' in report:
        scores.append(report['source_diversity']['diversity_score'])
    if 'trit_differentiation' in report:
        scores.append(report['trit_differentiation']['differentiation_score'])
    if 'geometric_coherence' in report:
        s = report['geometric_coherence'].get('coherence_score', 0)
        scores.append(max(0, s))  # может быть отрицательным при плохой согласованности
    if 'twilight_language' in report:
        scores.append(min(report['twilight_language']['twilight_rate'] * 10, 1.0))
    if 'hexagram_alignment' in report:
        scores.append(max(0, report['hexagram_alignment']['mean_alignment']))

    if scores:
        report['transmission_line_score'] = float(sum(scores) / len(scores))
        report['transmission_line_quality'] = (
            'strong'   if report['transmission_line_score'] > 0.6 else
            'moderate' if report['transmission_line_score'] > 0.3 else
            'weak'
        )

    return report


def print_report(report: Dict, indent: int = 0) -> None:
    """Красиво печатает отчёт."""
    prefix = '  ' * indent
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_report(value, indent + 1)
        elif isinstance(value, float):
            print(f"{prefix}{key}: {value:.4f}")
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            print(f"{prefix}{key}: {', '.join(value[:5])}")
        else:
            print(f"{prefix}{key}: {value}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute quality metrics beyond PPL for YiJing/NautilusMoME models'
    )
    parser.add_argument('--generated', type=str, nargs='+',
                        help='Generated text files (for twilight language analysis)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for report')
    args = parser.parse_args()

    generated_texts = []
    if args.generated:
        for path in args.generated:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    generated_texts.append(f.read())
                print(f'Loaded: {path}')
            except Exception as e:
                print(f'Warning: could not load {path}: {e}')

    if not generated_texts:
        # Демо: тест с примерами сумеречного языка
        generated_texts = [
            "Начало есть началость мысли. Единологится сквозь все слои."
            " The model чистость — мудрость, внутренняя степь познания.",
            "def compute_началость(self):\n    return self.единость * 0.7",
            "Чистота данных — основа. Данные есть данность бытия.",
        ]
        print("Using demo texts (no --generated specified)")

    report = compute_all_metrics(generated_texts=generated_texts)
    print("\n=== Quality Metrics Report ===")
    print_report(report)

    if args.output:
        # Преобразуем tensors в float для JSON-сериализации
        def jsonify(obj):
            if isinstance(obj, (torch.Tensor,)):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            return obj

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")

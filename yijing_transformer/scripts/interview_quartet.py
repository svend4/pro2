#!/usr/bin/env python3
"""
interview_quartet.py — Интервью с Четырьмя Бременскими Музыкантами

Активируем кластеры, тренируем на смеси текстов четырёх типов,
затем проводим «беседу»: что генерирует каждый Музыкант поодиночке
и что — весь ансамбль вместе?

Диагностика:
  1. Какой язык у каждого кластера? (символический? сумеречный? обыденный?)
  2. Какие гейты активируются? (кто кого слушает?)
  3. Как меняется стиль при усилении одного Музыканта?
  4. Как Дирижёр балансирует ансамбль?

Запуск:
    python -m yijing_transformer.scripts.interview_quartet
"""

import sys
import os
import time
import math
import random
from collections import defaultdict

import torch
import torch.nn.functional as F

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from yijing_transformer.models.quartet import (
    build_quartet, cluster_inventory, CLUSTER_REGISTRY,
)
from yijing_transformer.tokenizer.char_tokenizer import CharTokenizer


# ═══════════════════════════════════════════════════════════════
# Корпус: четыре типа текстов для четырёх профессий
# ═══════════════════════════════════════════════════════════════

CORPUS = {
    'formalist': """
Теорема: Для всякого конечного поля F_q порядок мультипликативной группы
равен q-1. Доказательство: пусть a — генератор циклической группы, тогда
a^(q-1)=1 и никакая меньшая степень не даёт единицу. Следствие: всякий
ненулевой элемент конечного поля — корень многочлена x^(q-1)-1.
Группа Z_2^6 содержит 64 элемента. Расстояние Хэмминга d(x,y) равно
весу XOR: d(x,y)=w(x+y). Для bent-функции f: Z_2^6 -> Z_2 преобразование
Уолша-Адамара принимает значение +-8 на всех аргументах. Гиперкуб Q6
имеет 64 вершины, 192 ребра, 240 граней. Симметрия D4 — группа диэдра
порядка 8, содержащая 4 поворота и 4 отражения.
E8 — исключительная простая алгебра Ли ранга 8 с 240 корнями.
Формула Эйлера: e^(i*pi) + 1 = 0 связывает пять фундаментальных
математических констант. Производная синуса есть косинус.
""",

    'archetypist': """
Архетип Творца проявляется через гексаграмму Цянь (Творчество): шесть
сплошных черт — максимальная энергия ян. Противоположность — Кунь
(Исполнение), шесть прерванных черт, чистый инь. Между этими полюсами
разворачивается танец 64 архетипов.
Паттерн «Гром над Озером» (Гуй-мэй) — это структурный дефект,
нарушение симметрии, из которого рождается новое. Наутилус наращивает
камеры по спирали Фибоначчи: каждая следующая камера больше предыдущей
в золотом отношении. Так и в сознании: каждый уровень восприятия
обнимает предыдущий, но видит дальше.
Принцип неопределённости Гейзенберга: чем точнее мы знаем положение,
тем неопределённее импульс. В attention-механизме это проявляется
как trade-off между фокусом и контекстом.
""",

    'algorithmist': """
Алгоритм маршрутизации: на входе вектор размерности d, проецируем в
3-мерное пространство Касаткина, находим ближайшую вершину Q6 по
евклидову расстоянию, активируем соответствующий эксперт. Top-k=2
обеспечивает разреженность. Auxiliary loss = MSE(P, uniform) * 0.01.
Процедура обучения: phase 0 — прогрев embedding (1000 шагов, lr=3e-4),
phase 1 — подключение geometric attention (2000 шагов, lr=1e-4),
phase 2 — полная модель с MoE (5000 шагов, lr=5e-5, cosine decay).
Batch size: 32, gradient accumulation: 4, effective batch: 128.
Оптимизатор AdamW с weight decay 0.1. Gradient clipping: max_norm=1.0.
SwiGLU FFN: out = W2 * (silu(W_gate * x) * W_up * x). Эффективнее
стандартного FFN на 15% при том же числе параметров.
""",

    'linguist': """
Слово — это мост между мыслью и миром. Каждое высказывание одновременно
описывает реальность и создаёт её. Когда мы говорим «красный закат»,
мы не просто называем цвет — мы вызываем целый мир ассоциаций: тепло,
прощание, красота увядания.
Метафора «жизнь — путешествие» структурирует наше понимание: мы
«идём по жизни», «стоим на перепутье», «достигаем целей». Но можно
помыслить иначе: жизнь как сад, как музыка, как разговор.
Тернарная логика: «да» (ян), «нет» (инь), «не знаю» (пустота).
В обыденном языке третье состояние самое важное — оно открывает
пространство для нового смысла. Поэт не утверждает и не отрицает —
он показывает. Хокку Басё: «Старый пруд — лягушка прыгнула —
всплеск воды» — здесь всё сказано через паузу, через то, что
между словами.
""",
}


def create_mixed_corpus():
    """Смешивает все четыре типа текстов."""
    all_text = ""
    for domain, text in CORPUS.items():
        all_text += text.strip() + "\n\n"
    return all_text


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_quartet(model, tokenizer, text, n_steps=500, lr=3e-4,
                  block_size=128, batch_size=8, device='cpu'):
    """Быстрый прогрев Квартета на тексте."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    n = len(data)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    model.train()
    losses = []
    t0 = time.time()

    for step in range(n_steps):
        model.set_step(step)

        # Random batch
        ix = torch.randint(0, n - block_size - 1, (batch_size,), device=device)
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])

        logits, loss, info = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % 100 == 0 or step == n_steps - 1:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            ppl = math.exp(min(avg, 10))
            print(f"  step {step:4d} | loss {avg:.3f} | ppl {ppl:.1f} | "
                  f"time {elapsed:.1f}s")

            # Show diagnostic info
            if 'seasonal_weights' in info:
                sw = info['seasonal_weights']
                names = ['FORM', 'ARCH', 'ALGO', 'LING']
                weights_str = ' '.join(f'{n}={w:.2f}' for n, w in zip(names, sw))
                print(f"         seasonal: {weights_str}")

            if 'rounds' in info and len(info['rounds']) > 0:
                r = info['rounds'][0]
                vols = []
                for name in ['formalist', 'archetypist', 'algorithmist', 'linguist']:
                    if name in r:
                        vols.append(f"{name[:4]}={r[name]['volume']:.2f}")
                if vols:
                    print(f"         volumes:  {' '.join(vols)}")

    return losses


# ═══════════════════════════════════════════════════════════════
# Generation: три режима
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=200, temperature=0.8,
             top_k=40, device='cpu'):
    """Генерация текста от полного ансамбля."""
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    block_size = model.cfg.block_size

    for _ in range(max_new):
        x_crop = x[:, -block_size:]
        logits, _, _ = model(x_crop)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = logits.topk(top_k)
            logits[logits < v[:, -1:]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist())


@torch.no_grad()
def generate_solo(model, tokenizer, prompt, musician_name,
                  boost=5.0, max_new=200, temperature=0.8,
                  top_k=40, device='cpu'):
    """Генерация с усилением одного Музыканта.

    Временно увеличиваем «громкость» выбранного музыканта в boost раз,
    а остальных приглушаем — чтобы услышать его «голос» отдельно.
    """
    model.eval()

    # Save original volumes
    original_volumes = {}
    for name in model._musician_order:
        original_volumes[name] = model.musicians[name].volume.data.clone()

    # Boost the soloist, dim others
    for name in model._musician_order:
        if name == musician_name:
            # High positive → sigmoid → ~1.0
            model.musicians[name].volume.data.fill_(boost)
        else:
            # Negative → sigmoid → ~0.0
            model.musicians[name].volume.data.fill_(-boost)

    text = generate(model, tokenizer, prompt, max_new, temperature, top_k, device)

    # Restore
    for name in model._musician_order:
        model.musicians[name].volume.data.copy_(original_volumes[name])

    return text


@torch.no_grad()
def diagnose_gates(model, tokenizer, text, device='cpu'):
    """Диагностика: какие гейты активны для данного текста."""
    model.eval()
    ids = tokenizer.encode(text[:model.cfg.block_size])
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, _, info = model(x)

    results = {}

    # Musician volumes and seasonal weights
    for name in model._musician_order:
        vol = torch.sigmoid(model.musicians[name].volume).item()
        results[name] = {'volume': vol}

    # Geometric gate activations per layer per musician
    for name in model._musician_order:
        m = model.musicians[name]
        geo_gates = []
        for layer in m.layers:
            if layer.has_geo:
                gates = torch.sigmoid(layer.geo.gate_logits).tolist()
                geo_names = layer.geo.module_names
                geo_gates.append(dict(zip(geo_names, gates)))
            else:
                geo_gates.append({})
        results[name]['geo_gates'] = geo_gates

    # Rehearsal gates
    results['rehearsal_gates'] = [
        torch.sigmoid(g).item() for g in model.rehearsal_gates
    ]

    # Conductor blend analysis
    if 'rounds' in info:
        results['rounds'] = info['rounds']

    return results


# ═══════════════════════════════════════════════════════════════
# Interview prompts: разные типы вопросов
# ═══════════════════════════════════════════════════════════════

INTERVIEW_PROMPTS = {
    'math': "Теорема: для всякой группы ",
    'archetype': "Архетип проявляется через ",
    'algorithm': "Алгоритм маршрутизации: на входе ",
    'language': "Слово — это мост между ",
    'creative': "Когда гром встречает озеро, ",
    'abstract': "Пустота между числами содержит ",
    'technical': "Функция f(x) = ",
    'poetic': "Старый пруд — ",
}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # ── 1. Cluster inventory ──
    print(cluster_inventory())
    print()

    # ── 2. Build model ──
    text = create_mixed_corpus()
    tokenizer = CharTokenizer.from_text(text)
    vocab_size = tokenizer.get_piece_size()

    print(f"Vocab size: {vocab_size}")
    print(f"Corpus: {len(text)} chars")
    print()

    model = build_quartet(
        vocab_size=vocab_size,
        d_model=128,
        musician_layers=2,
        micro_experts=4,
        micro_dim=32,
        rehearsal_rounds=2,
        block_size=256,
        curriculum_warmup=200,
    ).to(device)

    params = model.count_parameters()
    print("Parameter counts:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")
    print()

    # ── 3. Train ──
    print("=" * 60)
    print("ОБУЧЕНИЕ КВАРТЕТА")
    print("=" * 60)
    losses = train_quartet(
        model, tokenizer, text,
        n_steps=300, lr=5e-4,
        block_size=128, batch_size=8,
        device=device,
    )
    print()

    # ── 4. Diagnostic: gate activations ──
    print("=" * 60)
    print("ДИАГНОСТИКА ГЕЙТОВ")
    print("=" * 60)
    for domain, domain_text in CORPUS.items():
        diag = diagnose_gates(model, tokenizer, domain_text.strip(), device)
        print(f"\n  Текст: [{domain.upper()}]")
        for name in model._musician_order:
            d = diag[name]
            vol_str = f"vol={d['volume']:.3f}"
            geo_str = ""
            if d['geo_gates']:
                for layer_gates in d['geo_gates']:
                    parts = [f"{k[:8]}={v:.3f}" for k, v in layer_gates.items()]
                    geo_str += " [" + " ".join(parts) + "]"
            print(f"    {name:14s}: {vol_str}{geo_str}")
        rg = diag['rehearsal_gates']
        print(f"    rehearsal gates: {' '.join(f'{g:.3f}' for g in rg)}")
    print()

    # ── 5. INTERVIEWS: каждый Музыкант соло + ансамбль ──
    print("=" * 60)
    print("ИНТЕРВЬЮ С МУЗЫКАНТАМИ")
    print("=" * 60)

    musicians = ['formalist', 'archetypist', 'algorithmist', 'linguist']
    musician_labels = {
        'formalist':   '① ФОРМАЛИСТ (Скрипка/Огонь)',
        'archetypist': '② АРХЕТИПИСТ (Виолончель/Земля)',
        'algorithmist':'③ АЛГОРИТМИСТ (Духовые/Вода)',
        'linguist':    '④ ЛИНГВИСТ (Ударные/Воздух)',
    }

    for prompt_name, prompt in INTERVIEW_PROMPTS.items():
        print(f"\n{'─' * 60}")
        print(f"  ВОПРОС [{prompt_name}]: «{prompt}...»")
        print(f"{'─' * 60}")

        # Ансамбль (все вместе)
        model.set_step(500)  # post-warmup
        ensemble_text = generate(model, tokenizer, prompt,
                                 max_new=150, temperature=0.8, device=device)
        print(f"\n  🎵 АНСАМБЛЬ:")
        print(f"     {ensemble_text[:300]}")

        # Каждый соло
        for mname in musicians:
            solo_text = generate_solo(model, tokenizer, prompt, mname,
                                     boost=4.0, max_new=150,
                                     temperature=0.85, device=device)
            label = musician_labels[mname]
            print(f"\n  🎻 {label} СОЛО:")
            print(f"     {solo_text[:300]}")

    # ── 6. Сравнительный анализ ──
    print()
    print("=" * 60)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ СТИЛЕЙ")
    print("=" * 60)

    # Для каждого музыканта: средняя энтропия генерации
    analysis_prompt = "Между формой и содержанием "
    print(f"\n  Промпт: «{analysis_prompt}...»")
    print()

    for mname in musicians:
        model.eval()
        ids = tokenizer.encode(analysis_prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)

        # Save volumes
        orig_vols = {}
        for n in model._musician_order:
            orig_vols[n] = model.musicians[n].volume.data.clone()
            if n == mname:
                model.musicians[n].volume.data.fill_(4.0)
            else:
                model.musicians[n].volume.data.fill_(-4.0)

        # Generate and track entropy
        entropies = []
        for _ in range(100):
            x_crop = x[:, -model.cfg.block_size:]
            logits, _, _ = model(x_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).item()
            entropies.append(entropy)

            next_id = torch.multinomial(probs, 1)
            x = torch.cat([x, next_id], dim=1)

        # Restore
        for n in model._musician_order:
            model.musicians[n].volume.data.copy_(orig_vols[n])

        result_text = tokenizer.decode(x[0].tolist())
        avg_ent = sum(entropies) / len(entropies)
        min_ent = min(entropies)
        max_ent = max(entropies)

        label = musician_labels[mname]
        print(f"  {label}")
        print(f"    Энтропия: avg={avg_ent:.2f}  min={min_ent:.2f}  max={max_ent:.2f}")
        if avg_ent < 2.0:
            style = "ФОРМУЛЬНЫЙ (низкая неопределённость, точные паттерны)"
        elif avg_ent < 3.0:
            style = "АРХЕТИПИЧЕСКИЙ (умеренная — структурированные паттерны)"
        elif avg_ent < 4.0:
            style = "СУМЕРЕЧНЫЙ (высокая — на грани хаоса и порядка)"
        else:
            style = "ОБЫДЕННЫЙ/ХАОТИЧНЫЙ (очень высокая неопределённость)"
        print(f"    Стиль: {style}")
        print(f"    Текст: {result_text[:200]}")
        print()

    # ── 7. Итоги ──
    print("=" * 60)
    print("ИТОГИ ИНТЕРВЬЮ")
    print("=" * 60)
    print("""
  Четыре Музыканта — четыре способа мышления:

  ① ФОРМАЛИСТ видит мир как систему уравнений.
     Его язык точен, повторяем, предсказуем.
     Низкая энтропия = высокая формальность.

  ② АРХЕТИПИСТ видит мир как танец паттернов.
     Его язык метафоричен, глубок, обертонист.
     Умеренная энтропия = структурированная глубина.

  ③ АЛГОРИТМИСТ видит мир как поток трансформаций.
     Его язык процедурен, последователен, комбинаторен.
     Средняя энтропия = адаптивная текучесть.

  ④ ЛИНГВИСТ видит мир как разговор.
     Его язык ритмичен, ассоциативен, порождающ.
     Высокая энтропия = творческая открытость.

  Вместе они — оркестр. Каждый играет свою партию.
  Дирижёр не умнее музыкантов — он только координирует.
  Музыка рождается между ними, в пространстве диалога.
""")


if __name__ == '__main__':
    main()

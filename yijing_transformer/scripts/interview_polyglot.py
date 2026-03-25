#!/usr/bin/env python3
"""
interview_polyglot.py — Розеттский Камень: Четыре Языка Одной Истины

Тренируем PolyglotQuartet и показываем, как каждый музыкант
описывает одно и то же явление на СВОЁМ языке:

  ① ФОРМАЛИСТ  → E = m × c ^ 2
  ② АРХЕТИПИСТ → МАССА → ОГОНЬ × ОГОНЬ → ЭНЕРГИЯ
  ③ АЛГОРИТМИСТ → [m] ──→ [MUL] ──→ [E]
  ④ ЛИНГВИСТ   → «Масса, умноженная на квадрат скорости света...»

Запуск:
    python -m yijing_transformer.scripts.interview_polyglot
"""

import sys
import os
import time
import math

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from yijing_transformer.models.polyglot import (
    build_polyglot, PolyglotQuartet, VOCABS,
    FORMULA_VOCAB, ARCHETYPE_VOCAB, GRAPH_VOCAB,
)
from yijing_transformer.tokenizer.char_tokenizer import CharTokenizer


# ═══════════════════════════════════════════════════════════════
# Корпус
# ═══════════════════════════════════════════════════════════════

CORPUS = """
Теорема: Для всякого конечного поля порядок мультипликативной группы
равен q минус один. Энергия равна массе умноженной на квадрат скорости
света. Формула Эйлера связывает пять фундаментальных констант.
Архетип Творца проявляется через шесть сплошных черт — максимальная
энергия ян. Между полюсами разворачивается танец архетипов.
Наутилус наращивает камеры по спирали Фибоначчи.
Алгоритм маршрутизации: на входе вектор, проецируем в пространство,
находим ближайшую вершину, активируем эксперт. Оптимизатор AdamW
с weight decay. Gradient clipping max norm один.
Слово — это мост между мыслью и миром. Каждое высказывание одновременно
описывает реальность и создаёт её. Метафора структурирует понимание.
Тернарная логика: да, нет, не знаю. Третье состояние открывает
пространство для нового смысла.
Масса — это инерция, сопротивление движению. Энергия — способность
совершать работу. Свет — предельная скорость информации.
Гром над озером — структурный дефект, из которого рождается новое.
Функция преобразования: вход умножается на веса, проходит через
нелинейность, суммируется с bias. Выход подаётся на следующий слой.
""".strip()


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_polyglot(model, tokenizer, text, n_steps=500, lr=5e-4,
                   block_size=128, batch_size=8, device='cpu'):
    """Тренировка полиглот-квартета."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    n = len(data)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    model.train()
    losses = []
    t0 = time.time()

    for step in range(n_steps):
        model.set_step(step)

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

        if step % 50 == 0 or step == n_steps - 1:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            ppl = math.exp(min(avg, 10))

            ce = info.get('ce_loss', 0)
            ros = info.get('rosetta_loss', 0)
            sp = info.get('spec_loss', 0)

            print(f"  step {step:4d} | loss {avg:.3f} | ppl {ppl:.1f} | "
                  f"CE={ce:.3f} Ros={ros:.3f} Spec={sp:.3f} | {elapsed:.1f}s")

            if info.get('rosetta', {}).get('cosine_sims'):
                sims = info['rosetta']['cosine_sims']
                sim_str = ' '.join(f'{k}:{v:.3f}' for k, v in sims.items())
                print(f"         rosetta sims: {sim_str}")

            if info.get('rounds') and len(info['rounds']) > 0:
                blend = info['rounds'][-1].get('blend', [])
                if blend:
                    names = ['FORM', 'ARCH', 'ALGO', 'LING']
                    blend_str = ' '.join(f'{n}={w:.2f}' for n, w in zip(names, blend))
                    print(f"         blend: {blend_str}")

    return losses


# ═══════════════════════════════════════════════════════════════
# Розеттский Камень: показать все 4 языка
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def rosetta_stone(model, tokenizer, prompt, device='cpu'):
    """Для одного промпта показать вывод каждого музыканта на его языке."""
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    T = x.shape[1]
    if T > model.cfg.block_size:
        x = x[:, :model.cfg.block_size]
        T = model.cfg.block_size

    # Forward
    logits, _, info = model(x)

    results = {}

    # Лингвист: обычный текст (greedy decode от общих logits)
    text_ids = logits[0].argmax(dim=-1).tolist()
    results['linguist'] = tokenizer.decode(text_ids)

    # Формалист: декодируем spec_logits
    formalist_vocab = VOCABS['formalist']
    form_ids = info['spec_logits']['formalist'][0].argmax(dim=-1).tolist()
    results['formalist'] = formalist_vocab.decode_str(form_ids)

    # Архетипист
    arch_vocab = VOCABS['archetypist']
    arch_ids = info['spec_logits']['archetypist'][0].argmax(dim=-1).tolist()
    results['archetypist'] = arch_vocab.decode_str(arch_ids)

    # Алгоритмист
    algo_vocab = VOCABS['algorithmist']
    algo_ids = info['spec_logits']['algorithmist'][0].argmax(dim=-1).tolist()
    results['algorithmist'] = algo_vocab.decode_str(algo_ids)

    return results


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new=150,
                  temperature=0.8, top_k=40, device='cpu'):
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


# ═══════════════════════════════════════════════════════════════
# Анализ специализации
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_specialization(model, tokenizer, device='cpu'):
    """Анализ: насколько разнообразны «языки» четырёх музыкантов."""
    model.eval()

    test_texts = [
        "Энергия равна массе умноженной на квадрат скорости света",
        "Архетип Творца проявляется через максимальную энергию",
        "Алгоритм маршрутизации находит ближайшую вершину",
        "Слово это мост между мыслью и миром",
    ]

    musician_names = ['formalist', 'archetypist', 'algorithmist', 'linguist']
    vocabs = {
        'formalist': VOCABS['formalist'],
        'archetypist': VOCABS['archetypist'],
        'algorithmist': VOCABS['algorithmist'],
    }

    for text in test_texts:
        ids = tokenizer.encode(text[:model.cfg.block_size])
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _, _, info = model(x)

        print(f"\n  Текст: «{text[:60]}...»")

        for name in musician_names:
            spec_logits = info['spec_logits'][name][0]  # (T, V_spec)
            probs = F.softmax(spec_logits, dim=-1)

            # Средняя энтропия
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()

            # Топ-5 самых частых токенов
            avg_probs = probs.mean(dim=0)
            top5_vals, top5_idx = avg_probs.topk(5)

            if name in vocabs:
                top5_tokens = vocabs[name].decode(top5_idx.tolist())
            else:
                top5_tokens = [tokenizer.decode([i]) for i in top5_idx.tolist()]

            top5_str = ', '.join(
                f'"{t}"({v:.3f})' for t, v in zip(top5_tokens, top5_vals.tolist())
            )

            label = {'formalist': 'ФОРМ', 'archetypist': 'АРXЕ',
                      'algorithmist': 'АЛГО', 'linguist': 'ЛИНГ'}[name]
            print(f"    {label}: entropy={entropy:.2f}  top5=[{top5_str}]")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # ── 1. Prepare ──
    tokenizer = CharTokenizer.from_text(CORPUS)
    vocab_size = tokenizer.get_piece_size()
    print(f"Vocab: {vocab_size} chars, Corpus: {len(CORPUS)} chars")
    print(f"Specialized vocabs: formulas={len(FORMULA_VOCAB)}, "
          f"archetypes={len(ARCHETYPE_VOCAB)}, graphs={len(GRAPH_VOCAB)}")
    print()

    # ── 2. Build model ──
    model = build_polyglot(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        n_micro_experts=4,
        micro_dim=32,
        block_size=256,
        rehearsal_rounds=2,
        rosetta_weight=0.1,
        spec_weight=0.05,
    ).to(device)

    params = model.count_parameters()
    print("Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")
    print()

    # ── 3. Train ──
    print("=" * 65)
    print("  ОБУЧЕНИЕ ПОЛИГЛОТ-КВАРТЕТА")
    print("=" * 65)
    losses = train_polyglot(
        model, tokenizer, CORPUS,
        n_steps=400, lr=5e-4,
        block_size=128, batch_size=8,
        device=device,
    )
    print()

    # ── 4. РОЗЕТТСКИЙ КАМЕНЬ ──
    print("=" * 65)
    print("  РОЗЕТТСКИЙ КАМЕНЬ: Четыре Языка Одной Истины")
    print("=" * 65)

    prompts = [
        "Энергия равна массе ",
        "Архетип проявляется через ",
        "Алгоритм на входе ",
        "Слово это мост ",
        "Между формой и содержанием ",
    ]

    musician_labels = {
        'formalist':   '① ФОРМАЛИСТ  (формулы)',
        'archetypist': '② АРХЕТИПИСТ (архетипы)',
        'algorithmist':'③ АЛГОРИТМИСТ (графы)',
        'linguist':    '④ ЛИНГВИСТ   (слова)',
    }

    for prompt in prompts:
        print(f"\n{'─' * 65}")
        print(f"  ПРОМПТ: «{prompt}»")
        print(f"{'─' * 65}")

        results = rosetta_stone(model, tokenizer, prompt, device)

        for name in ['formalist', 'archetypist', 'algorithmist', 'linguist']:
            label = musician_labels[name]
            output = results[name][:200]
            print(f"\n  {label}:")
            print(f"    {output}")

    # ── 5. Генерация текста (ансамбль) ──
    print()
    print("=" * 65)
    print("  ГЕНЕРАЦИЯ ТЕКСТА (полный ансамбль)")
    print("=" * 65)

    gen_prompts = [
        "Энергия равна ",
        "Между числами ",
        "Функция преобразования ",
    ]

    for prompt in gen_prompts:
        text = generate_text(model, tokenizer, prompt,
                             max_new=120, temperature=0.8, device=device)
        print(f"\n  «{prompt}...»")
        print(f"  → {text[:250]}")

    # ── 6. Анализ специализации ──
    print()
    print("=" * 65)
    print("  АНАЛИЗ СПЕЦИАЛИЗАЦИИ")
    print("=" * 65)
    analyze_specialization(model, tokenizer, device)

    # ── 7. Rosetta диагностика ──
    print()
    print("=" * 65)
    print("  ROSETTA BRIDGE — Косинусные сходства")
    print("=" * 65)

    test_text = "Энергия равна массе умноженной на квадрат скорости света"
    ids = tokenizer.encode(test_text)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, _, info = model(x)

    if 'rosetta' in info and 'cosine_sims' in info.get('rosetta', {}):
        sims = info['rosetta']['cosine_sims']
    else:
        # Пересчитываем
        model.eval()
        tok = model.tok_emb(x)
        pos = model.pos_emb[:, :x.shape[1], :]
        shared = model.emb_drop(tok + pos)
        attn_mask = torch.triu(
            torch.ones(x.shape[1], x.shape[1], device=device, dtype=torch.bool),
            diagonal=1
        )
        hiddens = []
        for name in model._musician_order:
            _, h = model.musicians[name](shared, attn_mask=attn_mask)
            hiddens.append(h)
        _, rosetta_info = model.rosetta(hiddens)
        sims = rosetta_info['cosine_sims']

    names = ['ФОРМ', 'АРХЕ', 'АЛГО', 'ЛИНГ']
    print(f"\n  Текст: «{test_text}»\n")
    print(f"  {'':8s} {'ФОРМ':>8s} {'АРХЕ':>8s} {'АЛГО':>8s} {'ЛИНГ':>8s}")
    for i, ni in enumerate(names):
        row = f"  {ni:8s}"
        for j, nj in enumerate(names):
            if i == j:
                row += f" {'1.000':>8s}"
            else:
                key = f'{min(i,j)}-{max(i,j)}'
                val = sims.get(key, 0)
                row += f" {val:8.3f}"
        print(row)

    # ── 8. Итоги ──
    print()
    print("=" * 65)
    print("  ИТОГИ: Четыре Языка Одной Истины")
    print("=" * 65)
    print("""
  Один и тот же вход → четыре разных представления:

  ① ФОРМАЛИСТ  видит формулы и символы.
     Его язык: E, =, m, ×, c, ^, 2, +, ∫, ∑ ...
     Проверяемость: подставь числа — получи результат.

  ② АРХЕТИПИСТ видит стихии и соответствия.
     Его язык: МАССА→ЗЕМЛЯ, ЭНЕРГИЯ→ВОДА, СВЕТ→ОГОНЬ ...
     Проверяемость: каждый архетип = физический термин.

  ③ АЛГОРИТМИСТ видит графы и потоки.
     Его язык: INPUT──→TRANSFORM──→OUTPUT, SPLIT, MERGE ...
     Проверяемость: граф изоморфен вычислению.

  ④ ЛИНГВИСТ видит слова и метафоры.
     Его язык: обычный текст на естественном языке.
     Проверяемость: каждое слово ↔ научный термин.

  Розеттский мост (contrastive loss) обеспечивает,
  что все четверо описывают ОДНО И ТО ЖЕ явление.
  Cosine similarity → 1.0 = полное семантическое согласие.
""")

    final_loss = sum(losses[-20:]) / len(losses[-20:])
    print(f"  Final loss: {final_loss:.3f}")
    print(f"  Final perplexity: {math.exp(min(final_loss, 10)):.1f}")
    print()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Twilight Language Re-test — перепроверка Сумеречного языка после Stage A.

Воспроизводит ТОЧНО те же промпты из CROSS_DOMAIN_CONCEPT.md и deep_interview,
чтобы сравнить: изменился ли Сумеречный язык после изменений в коде.
"""

import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import sentencepiece as spm

from scripts.train_nautilus_mome import NautilusMoME, CHECKPOINT_PATH, TOKENIZER_MODEL
from inference.generate import generate as standard_generate
from inference.domain_locked_generate import domain_locked_generate


@torch.no_grad()
def full_analysis(model, sp, text):
    """Get routing, archetype, synth, twilight for text."""
    model.eval()
    tokens = sp.encode(text)
    if len(tokens) < 2:
        return {}
    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
    logits, _, info = model(idx)

    # PPL
    shift_logits = logits[:, :-1, :]
    targets = idx[:, 1:]
    loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), targets.reshape(-1))
    ppl = math.exp(min(loss.item(), 20))

    # Routing
    routing = info['routing'][0]
    avg_routing = routing.mean(dim=0)
    names = model.EXPERT_NAMES[:model.n_experts]
    routing_dict = {names[i]: avg_routing[i].item() for i in range(len(names))}
    top_expert = max(routing_dict, key=routing_dict.get)

    # Confidence
    routing_probs = F.softmax(routing, dim=-1)
    entropy = -(routing_probs * (routing_probs + 1e-10).log()).sum(dim=-1)
    confidence = 1.0 - entropy.mean().item() / math.log(model.n_experts)

    # Domain switching
    dom = routing.argmax(dim=-1)
    switches = (dom[1:] != dom[:-1]).sum().item()
    mixing = switches / max(len(tokens) - 1, 1)

    # Archetype
    arch = info.get('archetype', {})
    # SYNTH
    synth = info.get('synth', {})
    # Twilight
    twi = info.get('twilight', {})

    return {
        'ppl': ppl,
        'routing': routing_dict,
        'top_expert': top_expert,
        'confidence': confidence,
        'mixing': mixing,
        'arch_code': arch.get('top_archetype', '?'),
        'arch_name': arch.get('top_name', '?'),
        'arch_prob': arch.get('top_prob', 0),
        'synth_frac': synth.get('activation_frac', 0),
        'synth_gate': synth.get('gate', 0),
        'twi_strength': twi.get('twilight_strength', 0),
        'twi_blend': twi.get('blend_ratio', 0),
    }


def gen_both(model, sp, prompt, max_tokens=100, temp=0.8):
    """Generate with standard and domain-locked, return both."""
    text_std = standard_generate(model, sp, prompt, max_tokens=max_tokens, temperature=temp)
    text_dl, stats_dl = domain_locked_generate(model, sp, prompt, max_tokens=max_tokens, temperature=temp)
    return text_std, text_dl, stats_dl


def print_section(title):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def print_analysis(a, prefix="  "):
    top3 = sorted(a['routing'].items(), key=lambda x: -x[1])[:3]
    routing_str = " | ".join(f"{n}={v:.3f}" for n, v in top3)
    print(f"{prefix}PPL={a['ppl']:.1f}  top={a['top_expert']}  conf={a['confidence']*100:.1f}%  mix={a['mixing']*100:.0f}%")
    print(f"{prefix}Routing: {routing_str}")
    print(f"{prefix}Archetype: {a['arch_code']} ({a['arch_name']}) p={a['arch_prob']:.3f}")
    print(f"{prefix}SYNTH: frac={a['synth_frac']:.2f} gate={a['synth_gate']:.3f}")
    print(f"{prefix}Twilight: strength={a['twi_strength']:.3f} blend={a['twi_blend']:.3f}")


def main():
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location='cpu')
    args = ckpt.get('args', {})
    model = NautilusMoME(
        vocab_size=args.get('vocab_size', sp.get_piece_size()),
        d_model=args.get('d_model', 128),
        n_layers=args.get('n_layers', 4),
        n_heads=args.get('n_heads', 4),
        block_size=args.get('block_size', 256),
        d_expert=args.get('d_expert', 128),
        n_experts=args.get('n_experts', 6),
        top_k=args.get('top_k', 2),
    )
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    print(f"Loaded: step {ckpt.get('step', '?')}, {sum(p.numel() for p in model.parameters()):,} params")

    # ═══════════════════════════════════════════════════════
    # PART 1: Exact prompts from CROSS_DOMAIN_CONCEPT.md
    # ═══════════════════════════════════════════════════════

    print_section("PART 1: ДИАЛОГ — те же 6 вопросов из CROSS_DOMAIN_CONCEPT.md")

    dialogue_prompts = [
        ("Кто ты?", "# О себе\n\nЯ — нейросеть, которая "),
        ("Что внутри тебя?", "# Архитектура\n\nВнутри меня находятся "),
        ("Когда я вижу стихи...", "# Анализ\n\nКогда я вижу стихи, я "),
        ("Однажды я смогу", "# Будущее\n\nОднажды я смогу "),
        ("Будущее нейросетей", "# Прогноз\n\nБудущее нейросетей — это "),
        ("Моя мечта", "# Мечта\n\nМоя мечта — это "),
    ]

    for title, prompt in dialogue_prompts:
        print(f"\n  ┌─ Вопрос: {title}")
        print(f"  │ Промпт: {prompt.strip()}")

        text_std, text_dl, dl_stats = gen_both(model, sp, prompt, max_tokens=80, temp=0.8)

        a_std = full_analysis(model, sp, prompt + text_std)
        a_dl = full_analysis(model, sp, prompt + text_dl)

        print(f"  │")
        print(f"  │ STANDARD: {text_std[:200]}")
        print_analysis(a_std, prefix="  │   ")
        print(f"  │")
        print(f"  │ DOMAIN-LOCKED ({dl_stats['dominant_expert']}): {text_dl[:200]}")
        print_analysis(a_dl, prefix="  │   ")
        print(f"  └─")

    # ═══════════════════════════════════════════════════════
    # PART 2: Twilight Language Triggers
    # ═══════════════════════════════════════════════════════

    print_section("PART 2: СУМЕРЕЧНЫЙ ЯЗЫК — триггеры неологизмов")

    twilight_prompts = [
        ("Übermensch / Заратустра", "Так говорил Заратустра: хаос рождает "),
        ("Кристалл + Аксиома", "Кристалл + Аксиома → "),
        ("Обучение = эволюция", "Обучение — это эволюция. Мой архетип — "),
        ("Аналогия = мост", "Аналогия — это мост между "),
        ("Machine language", "The language of the machine is not words but "),
        ("Формула реальности", "Все формулы описывают одну реальность "),
        ("Невостность", "Невостность души означает "),
        ("Поэтиналогия", "Поэтиналогикупенчивость закономер"),
    ]

    for title, prompt in twilight_prompts:
        print(f"\n  ┌─ {title}")
        print(f"  │ Промпт: {prompt}")

        text_std, text_dl, dl_stats = gen_both(model, sp, prompt, max_tokens=60, temp=0.9)
        a_std = full_analysis(model, sp, prompt + text_std)
        a_dl = full_analysis(model, sp, prompt + text_dl)

        print(f"  │ STANDARD: {text_std[:200]}")
        print(f"  │   TWI={a_std['twi_strength']:.3f}  SYNTH={a_std['synth_frac']:.2f}  "
              f"top={a_std['top_expert']}  arch={a_std['arch_code']} ({a_std['arch_name']})")
        print(f"  │ LOCKED ({dl_stats['dominant_expert']}): {text_dl[:200]}")
        print(f"  │   TWI={a_dl['twi_strength']:.3f}  SYNTH={a_dl['synth_frac']:.2f}  "
              f"top={a_dl['top_expert']}  arch={a_dl['arch_code']} ({a_dl['arch_name']})")
        print(f"  └─")

    # ═══════════════════════════════════════════════════════
    # PART 3: Code vs Twilight (контроль)
    # ═══════════════════════════════════════════════════════

    print_section("PART 3: КОНТРОЛЬ — код vs Сумеречный (должны быть разные TWI)")

    control_prompts = [
        ("Pure Python (expect TWI≈0)", "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        "),
        ("Pure imports (expect TWI≈0)", "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.data import DataLoader\n"),
        ("Russian philosophy (expect TWI>0.5)", "# Философия\n\nВсе формулы описывают одну реальность разными словами. Законости "),
        ("Zarathustra ML (expect TWI>0.8)", "# Заратустра и нейросети\n\nХаос порождает танцующую звезду. Dropout порождает "),
    ]

    for title, prompt in control_prompts:
        text_std = standard_generate(model, sp, prompt, max_tokens=60, temperature=0.7)
        a = full_analysis(model, sp, prompt + text_std)
        print(f"\n  {title}")
        print(f"    Output: {text_std[:150]}")
        print(f"    TWI={a['twi_strength']:.3f}  SYNTH={a['synth_frac']:.2f}  "
              f"top={a['top_expert']}  PPL={a['ppl']:.1f}  arch={a['arch_code']} ({a['arch_name']})")

    # ═══════════════════════════════════════════════════════
    # PART 4: Самые интересные слова
    # ═══════════════════════════════════════════════════════

    print_section("PART 4: ГЕНЕРАЦИЯ НЕОЛОГИЗМОВ — 5 попыток по 3 стратегии")

    neologism_seed = "Новое слово для описания процесса обучения нейросети: "

    print(f"\n  Seed: {neologism_seed}")
    for i in range(5):
        t_std = standard_generate(model, sp, neologism_seed, max_tokens=30, temperature=1.0 + i*0.1)
        t_dl, _ = domain_locked_generate(model, sp, neologism_seed, max_tokens=30, temperature=1.0 + i*0.1)
        t_mir, _ = domain_locked_generate(model, sp, neologism_seed, max_tokens=30,
                                           temperature=0.9, use_mirostat=True, mirostat_tau=6.0)
        print(f"\n  Run {i+1} (T={1.0+i*0.1:.1f}):")
        print(f"    Standard:  {t_std[:120]}")
        print(f"    Locked:    {t_dl[:120]}")
        print(f"    Mirostat:  {t_mir[:120]}")


if __name__ == '__main__':
    main()

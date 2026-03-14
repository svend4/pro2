"""Тестовый диалог с обученной моделью."""
import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yijing_transformer'))

from config.config import YiJingConfig
from models.model import YiJingGPT

CHECKPOINT = "checkpoints/checkpoint_step_4000.pt"
DEVICE = torch.device("cpu")

print(f"Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
cfg = ckpt['config']
model = YiJingGPT(cfg).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Step: {ckpt['step']}, vocab_size={cfg.vocab_size}, d_model={cfg.d_model}, n_layers={cfg.n_layers}")

try:
    from tokenizer.tokenizer_utils import load_tokenizer
    tokenizer = load_tokenizer()
    print("Tokenizer: SentencePiece")
except Exception:
    from tokenizer.char_tokenizer import CharTokenizer
    tokenizer = CharTokenizer()
    print("Tokenizer: char-level")


@torch.no_grad()
def generate(prompt, max_tokens=150, temperature=0.8, top_k=50, top_p=0.9, rep_penalty=1.2):
    ids = tokenizer.encode(prompt)
    if not ids:
        return "<пустой токен>"
    context = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    generated = []
    for _ in range(max_tokens):
        idx_cond = context[:, -cfg.block_size:]
        logits, _, _ = model(idx_cond)
        logits = logits[0, -1, :].clone()
        past = context[0, -50:].tolist()
        for t in set(past):
            if t < logits.size(0):
                logits[t] /= rep_penalty
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        if top_k > 0:
            v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
            probs[probs < v[-1]] = 0.0
        if 0 < top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            cs = torch.cumsum(sp, dim=-1)
            mask = cs > top_p
            mask[1:] = mask[:-1].clone()
            mask[0] = False
            probs[si[mask]] = 0.0
        probs = probs / probs.sum()
        nxt = torch.multinomial(probs, 1)
        context = torch.cat((context, nxt.unsqueeze(0)), dim=1)
        tid = nxt.item()
        eos = tokenizer.eos_id() if hasattr(tokenizer, 'eos_id') else -1
        if eos != -1 and tid == eos:
            break
        generated.append(tid)
    return tokenizer.decode(generated)


# Диалог
prompts = [
    "The transformer model",
    "In the beginning",
    "Language models can",
    "The hexagrams of the Yi Jing represent",
    "Once upon a time there was a",
]

print("\n" + "="*60)
print("ДИАЛОГ С МОДЕЛЬЮ (checkpoint_step_4000.pt)")
print("="*60)

for prompt in prompts:
    print(f"\n[ВОПРОС]: {prompt}")
    response = generate(prompt, max_tokens=120, temperature=0.85, top_k=50, top_p=0.9)
    print(f"[МОДЕЛЬ]: {prompt}{response}")
    print("-"*60)

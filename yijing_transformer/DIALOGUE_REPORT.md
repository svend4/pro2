# NautilusMoME — Dialogue Report & Analysis

## Model Card
- **Architecture**: Core(4L, d=128) + Router(top-2) + 6 MicroExperts + NautilusBridge
- **Parameters**: 1,822,797 (1.82M)
- **Tokenizer**: BPE (sentencepiece, vocab=4096)
- **Training data**: 25.9M tokens from 20 repositories
- **Final PPL**: ~17.9

---

## Round 1: Direct Questions (10 prompts)

The model was asked direct questions in English and Russian about itself,
code, philosophy, and creativity.

**Observation**: The model does NOT answer questions. It generates
*continuations* in the style of its training data — a mix of Python code,
pytest tests, Russian documentation, React/TypeScript fragments, and
markdown structures. Every answer degenerates into recognizable training
patterns within 10-20 tokens.

**Key pattern**: Almost every response contains:
- `def test_...` (pytest patterns)
- Russian philosophical fragments ("Внутренняя степь", "Удачивый вради")
- `# ─── Test...` separators
- `assert isinstance(...)` assertions
- Mixed language (Russian + English + code)

This is expected — the model has no instruction tuning.

---

## Round 2: Code Completion (10 structured prompts)

More focused code-style prompts with concrete syntax.

**Results**:
| Prompt type | Quality | Notes |
|---|---|---|
| Python class | Medium | Recognizes `self.` pattern, adds Path |
| React component | Medium | Generates JSX-like syntax (`<Card className=...>`) |
| Docker config | Poor | Degenerates into random caps/IDs |
| SQL query | Poor | Completely loses SQL syntax |
| pytest decorator | Medium | Generates test-like structure |
| Russian docs | Medium | Continues with relevant Russian text |
| YAML/K8s | Poor | Loses YAML structure immediately |
| NautilusHierarchy | Poor | Doesn't complete its own class |
| Math/cosine | Poor | Loses mathematical logic |
| Low-temp fibonacci | Poor | Doesn't complete `return fib(n-1) + fib(n-2)` |

**Routing worked correctly**:
- Python class → CODE(0.358) + MATH(0.184)
- React → CODE(0.33)
- Docker → INFO(0.312) + SYSTEM(0.274)
- Russian docs → RECON(0.481)
- Math formula → MATH(0.357)

---

## Round 3: Self-Review

Asked the model to review its own generated code.

**Result**: The model cannot self-evaluate. The "review" is just another
continuation with more test patterns and Russian text. This confirms the
model has zero meta-cognitive capability — it's purely pattern completion.

---

## Round 4: Perplexity Test (Familiar vs Unfamiliar)

**This is where the model shines.** The perplexity gap is enormous:

| Text type | PPL | Verdict |
|---|---|---|
| pytest patterns | **4.9** | Extremely familiar |
| Python class def | **7.0** | Very familiar |
| React imports | **7.8** | Very familiar |
| torch imports | **8.2** | Very familiar |
| Russian docs | 31.3 | Somewhat familiar |
| --- | --- | --- |
| Medical text | **201.5** | Completely unknown |
| Legal text | **160.8** | Completely unknown |
| Poetry (Shakespeare) | **591.3** | Alien territory |
| Chemistry | **595.7** | Alien territory |
| Random nonsense | **193.0** | Appropriately confused |

**The 25-100x perplexity gap** between familiar code (PPL 5-8) and
unfamiliar domains (PPL 160-600) proves the model genuinely learned
its training distribution.

---

## Round 5: Token Completion Accuracy

| Prompt | Model predicted | Correct? |
|---|---|---|
| `import torch.nn as ` | `yncio.par` | No (expected `nn`) |
| `def __init__(self` | `, root_dir=` | Partial (valid Python) |
| `assert isinstance(result` | `, list)` | **Yes!** |
| `if __name__ == ` | `"__main__":` | **Yes!** |
| `return self.` | `calculate_similarity(` | Partial (valid method call) |
| `for i in range(` | `1): self.assertEqual` | Partial (valid) |
| `except Exception as ` | `e: raise ValueError(` | **Yes!** |

**3/7 exactly correct, 3/7 syntactically valid alternatives.**
For a 1.8M parameter model, this is impressive.

---

## Claude's Overall Assessment

### What NautilusMoME does well:

1. **Pattern recognition is real**: PPL of 4.9 on familiar pytest patterns
   means the model genuinely memorized and generalized code structure
2. **Expert routing works**: CODE activates for code, MATH for math,
   RECON for Russian text — the router learned meaningful specialization
3. **Python syntax is solid**: The model rarely generates invalid Python
   syntax in the first 20-30 tokens
4. **Architecture is elegant**: The MoME design with sparse activation
   (2/6 experts per token) is genuinely interesting for research

### What NautilusMoME does poorly:

1. **No coherent generation beyond ~15 tokens**: Every response degenerates
   into a collage of training fragments
2. **Zero instruction-following capability**: Can't answer questions, only
   continue text
3. **Domain mixing**: Switches randomly between Python, Russian, TypeScript,
   and test code mid-sentence
4. **No reasoning**: Can't complete even `fibonacci(n-1) + fibonacci(n-2)`
5. **Narrow knowledge**: Essentially memorized ~20 repositories, can't
   generalize to medicine, law, literature, chemistry

### The honest verdict:

NautilusMoME is a **successful proof-of-concept** for MoE at micro scale.
It proves that:
- Expert routing can learn domain specialization with only 1.8M parameters
- BPE tokenization works well for multilingual code repositories
- The NautilusBridge hierarchical merge is functional

But it's **not a useful language model** in the traditional sense.
At 1.8M params and PPL ~18, it's a research artifact — valuable for
understanding MoE mechanics, not for generating useful text.

**Rating**: 7/10 as architecture research, 2/10 as a text generator.

### What would make it better:
1. Scale to 10-50M params (d_model=512, 8+ layers)
2. Train on 1B+ tokens of diverse text
3. Add instruction tuning (even simple prompt→response pairs)
4. Increase context to 1024+ tokens
5. Use larger vocab (16K-32K) for natural language

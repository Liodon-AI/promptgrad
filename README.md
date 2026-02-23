# promptgrad

**Prompt Sensitivity Analyzer** — measure how robust your LLM prompts are to linguistic perturbations.

[![PyPI version](https://badge.fury.io/py/promptgrad.svg)](https://pypi.org/project/promptgrad/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Liodon-AI/promptgrad/actions/workflows/ci.yml/badge.svg)](https://github.com/Liodon-AI/promptgrad/actions)

---

Prompt engineering is chaos. The same instruction, phrased slightly differently, can produce wildly different outputs. **promptgrad** quantifies this instability.

Given a prompt, it:
1. **Perturbs** it with six linguistic strategies (synonym swap, word deletion, paraphrase, casing, punctuation, sentence reorder)
2. **Embeds** every variant using TF-IDF (built-in, zero dependencies) or semantic models
3. **Computes** cosine similarity shifts, output variance, and Shannon entropy
4. **Detects** unstable instructions and surfaces which tokens drive the instability
5. **Outputs** a robustness score (0–1), a per-token sensitivity heatmap, and a strategy-level radar chart

---

## Installation

```bash
# Minimal install (TF-IDF embeddings, no ML dependencies)
pip install promptgrad

# With local semantic embeddings (recommended)
pip install "promptgrad[local]"

# With OpenAI embeddings
pip install "promptgrad[openai]"

# With visualisation
pip install "promptgrad[viz]"

# Everything
pip install "promptgrad[all]"
```

---

## Quick Start

```python
from promptgrad import PromptAnalyzer

analyzer = PromptAnalyzer()

report = analyzer.analyze("Summarize the document in three sentences.")

print(report.robustness_score)      # e.g. 0.847
print(report.stability_label)       # "STABLE"
print(report.top_sensitive_tokens()) # [("Summarize", 0.91), ("three", 0.72), ...]

for warning in report.warnings:
    print(warning)
```

### Visualise

```python
analyzer.plot(report)                 # All four charts
analyzer.plot(report, kind="heatmap") # Just the token heatmap
analyzer.plot(report, kind="gauge")   # Just the robustness gauge
analyzer.plot(report, save_path="report.png", show=False)
```

### Compare multiple prompts

```python
result = analyzer.compare([
    "Summarize the document in three sentences.",
    "Please provide a summary of the document.",
    "What are the key points in this document?",
])

for prompt, score in result["ranked"]:
    print(f"{score:.3f}  {prompt}")
```

### CLI

```bash
# Analyse a prompt
promptgrad analyze "List the top 5 programming languages."

# With semantic embeddings and save plot
promptgrad analyze "Explain quantum computing." \
    --backend sentence_transformers \
    --plot report.png

# Export JSON report
promptgrad analyze "Write a haiku about rain." --json report.json

# Compare prompts from a file (one per line)
promptgrad compare prompts.txt
```

---

## Outputs

| Field | Type | Description |
|---|---|---|
| `robustness_score` | `float` | 0.0 (unstable) → 1.0 (robust) |
| `stability_label` | `str` | `VERY STABLE` / `STABLE` / `FRAGILE` / `UNSTABLE` |
| `mean_cosine_similarity` | `float` | Average similarity to original across all perturbations |
| `embedding_shift_std` | `float` | Standard deviation of L2 shifts |
| `entropy` | `float` | Shannon entropy of the similarity distribution |
| `output_variance` | `float \| None` | Variance of LLM output embeddings (if provided) |
| `token_importance` | `dict[str, float]` | Token → sensitivity score (0–1) |
| `per_strategy` | `dict[str, float]` | Mean cosine similarity per perturbation type |
| `warnings` | `list[str]` | Human-readable instability warnings |

---

## Embedding Backends

| Backend | Requires | Quality | Use when |
|---|---|---|---|
| `tfidf` | nothing | lexical only | Fast tests, CI, offline |
| `sentence_transformers` | `pip install promptgrad[local]` | semantic | Recommended default |
| `openai` | `pip install promptgrad[openai]` + API key | semantic | Production / highest quality |

```python
# Auto-selects sentence_transformers if installed, else tfidf
analyzer = PromptAnalyzer(embedding_backend="auto")

# Explicit
analyzer = PromptAnalyzer(embedding_backend="sentence_transformers",
                          model_name="all-MiniLM-L6-v2")

analyzer = PromptAnalyzer(embedding_backend="openai",
                          model="text-embedding-3-small",
                          api_key="sk-...")
```

---

## Perturbation Strategies

| Strategy | What it does |
|---|---|
| `synonym_substitution` | Swaps words with near-synonyms |
| `word_deletion` | Removes individual non-stopword tokens |
| `paraphrase` | Structural rewrites (imperative ↔ question, etc.) |
| `casing_variation` | Title, UPPER, lower, mIxEd case |
| `punctuation_variation` | Adds, removes, or changes terminal punctuation |
| `order_shuffle` | Shuffles sentence order in multi-sentence prompts |

Use all of them (default) or pick a subset:

```python
analyzer = PromptAnalyzer(perturbations=["synonym_substitution", "word_deletion"])
```

Bring your own:

```python
from promptgrad import Perturbation, PerturbationResult

class BackTranslation(Perturbation):
    name = "back_translation"

    def apply(self, prompt, n=5, seed=None):
        # ... call translation API ...
        return [PerturbationResult(original=prompt, perturbed=translated, strategy=self.name)]

analyzer = PromptAnalyzer(perturbations=[BackTranslation()])
```

---

## Measuring Output Variance

If you have LLM responses for the perturbed prompts, pass them in to compute semantic output variance:

```python
import openai

client = openai.OpenAI()

def complete(prompt):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

# Generate outputs for each perturbed variant
from promptgrad.perturbations import SynonymSubstitution
perturbed = SynonymSubstitution().apply(my_prompt, n=10)
outputs = [complete(r.perturbed) for r in perturbed]

report = analyzer.analyze(my_prompt, output_texts=outputs)
print(report.output_variance)
```

---

## Development

```bash
git clone https://github.com/Liodon-AI/promptgrad
cd promptgrad
pip install -e ".[dev]"
pytest -v
```

---

## License

MIT © [Liodon AI](https://github.com/Liodon-AI)

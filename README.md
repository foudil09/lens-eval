# lens-eval

[![PyPI](https://img.shields.io/pypi/v/lens-eval.svg)](https://pypi.org/project/lens-eval/)
[![Python](https://img.shields.io/pypi/pyversions/lens-eval.svg)](https://pypi.org/project/lens-eval/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**An interpretable, *learned* text-quality metric — for RAG, machine translation,
and dialogue.**

`lens-eval` scores text along four transparent dimensions — **semantic**, **nli**,
**naturalness**, **emotion** — and *learns from human judgments* how to weigh them for the job
at hand. The same machinery re-ranks **RAG answers**, scores **machine translation**, and rates
**dialogue** — and it always reports the coefficients, so you can see *why* a
text scored the way it did. It's small, runs on both CPU and GPU, and is fully interpretable.

A clear use case is **RAG**: you retrieved a passage, generated a few candidate answers,
which one do you serve? Assemble the grounded prompt your pipeline already builds, then let a
pretrained metric rank the answers against it. No training, no LLM judge:

```python
from lens_eval import LENS

prompt = f"Using the following guideline, answer the question.\nGuideline: {context}\nQuestion: {question}"

metric = LENS.load("foudil/lens_focus")          # pretrained, from the Hugging Face Hub
scores = metric.score(candidate_answers, references=[prompt] * len(candidate_answers))
```

```
Question:   How much paracetamol can I take for a 39°C fever?
Retrieved Context:  For adults, paracetamol 500–1000 mg every 4–6 hours, max 4 g/day.

Candidate answers, ranked by lens_focus (higher = better grounded answer):
  0.88  Based on the guideline, an adult can take 500–1000 mg every 4–6 hours, max 4 g/day.
  0.39  I'm not a doctor, so I can't say anything about doses.                 ← evasive
  0.25  You should take 2000 mg every two hours until the fever is gone.       ← wrong dose
  0.03  A fever can be uncomfortable. Make sure to rest and stay hydrated.     ← off-topic
```

The grounded answer wins decisively and the off-topic one sinks. `lens_focus` was trained on
RewardBench-2's *Focus* task and outranks many strong LLM-as-a-judge models on it — at a fraction of
the cost. Full script: [`examples/rag_rerank.py`](examples/rag_rerank.py).

<p align="center">
  <img src="https://raw.githubusercontent.com/foudil09/lens-eval/main/assets/rewardbench2_focus.png"
       width="600" alt="RewardBench-2 Focus: LENS vs. 16 LLM-as-a-judge models">
  <br><sub>A ~0.6 B-parameter LENS metric outranks most LLM-as-a-judge models on RewardBench-2 · Focus — including ones orders of magnitude larger.</sub>
</p>

## Install

```bash
pip install 'lens-eval[encoders]'  # score raw text (transformer encoders included)
pip install lens-eval              # core only: combiner + your own feature matrix
pip install 'lens-eval[all]'       # everything (EBM/GBM combiners, HTML reports, ...)
```

The core install is intentionally light (numpy / scipy / scikit-learn). The `encoders` extra
adds the transformer encoders that turn raw text into features; skip it if you already have a
feature matrix.

## Pretrained metrics on the Hub

Load any of these with `LENS.load("owner/repo")` — no fitting required:

| Metric | Task | Score from text with |
|---|---|---|
| [`foudil/lens_focus`](https://huggingface.co/foudil/lens_focus) | RAG / best-answer selection (RewardBench-2 Focus) | `score(answers, references=[prompt]*n)` |
| [`foudil/lens_wmt_da`](https://huggingface.co/foudil/lens_wmt_da) | machine-translation quality (WMT DA) | `score(translations, references=refs)` |
| `foudil/lens_uses_knowledge`, `lens_engaging`, `lens_maintains_context`, `lens_understandable`, `lens_natural` | knowledge-grounded dialogue facets (TopicalChat-USR) | combine several reference signals — see [DOCS](DOCS.md) |

## Train your own

Have human ratings for your own domain? Fit a combiner on them. You need **at least 50 rows**.

```python
from lens_eval import LENS

# Candidate texts, their references, and human quality scores.
hyps = ["the cat sat on the mat", ...]   
refs = ["a cat is on the mat", ...]
y    = [4.0, ...]                             # e.g. 1–5 Likert, or 0–100 DA scores

lens = LENS().fit(texts=hyps, references=refs, scores=y)

scores = lens.score(new_hyps, references=new_refs)   # array of quality scores
order  = lens.rank(candidates, references=refs)       # best-first indices
delta  = lens.compare(a, b, references=refs)          # >0 means `a` is better

lens.report()                                         # what it learned + diagnostics
lens.save("./my-lens")
```

Running on a GPU? `from lens_eval import configure; configure(device="cuda")` first.

## What it does

- **Three task modes, auto-detected** from the targets you pass:
  - `scores=` → **regression** (continuous, bounded, ordinal, or binary)
  - `pairs=`  → **pairwise** (winner/loser index pairs)
  - `ranks=` + `groups=` → **ranking**
- **Auto model selection.** Cross-validates a ladder of combiners (`glm` → `glm_interactions`
  → `ebm` → `gbm`), gated by sample size, and keeps the simplest one within 1 SE of the best.
- **Interpretable by construction.** Coefficients, drop-column ablation, per-dimension
  contributions, and a text or HTML report (`lens.report_html("report.html")`).
- **Bring your own features.** Pass `features=` (a NumPy array or DataFrame) to skip encoding —
  useful when your dimension scores already live in a CSV, or come from custom reference signals.
- **Save, load, and share** locally or via the Hugging Face Hub.

## Benchmarks

**Machine translation (WMT MQM).** On the WMT MQM segment- and system-level pairwise-accuracy
benchmarks, LENS reaches specialized-metric quality — matching COMET-22 and BLEURT-20 and
approaching the 11–13 B-parameter xCOMET-XXL and MetricX — while *learning only ≤5 parameters*
(the linear combiner) on top of frozen, off-the-shelf encoders.

<p align="center">
  <img src="https://raw.githubusercontent.com/foudil09/lens-eval/main/assets/parameter_efficiency.png"
       width="900" alt="WMT MQM pairwise accuracy vs. learned parameters: LENS reaches specialized-metric quality with ≤5 learned parameters">
</p>

**The cost is negligible.** The x-axis above counts *learned* parameters, where LENS uses ≤5.
Its full inference footprint is ~0.6 B parameters — the four frozen, off-the-shelf encoders, with
no task-specific fine-tuning — which is the same ballpark as COMET-22, roughly **20× smaller** than
xCOMET-XXL / MetricX, and **~100× smaller** than the 70 B+ LLM judges it outranks on Focus. It runs
on CPU, trains in seconds, and the learned part stays a handful of interpretable coefficients.

## Command line

```bash
lens-eval fit   --texts hyp.txt --refs ref.txt --scores scores.csv --output ./my-lens
lens-eval score --model ./my-lens --texts new.txt --refs ref.txt --output preds.csv
lens-eval report ./my-lens --html report.html
```

`--texts`/`--refs` take one item per line (`.txt`) or the first column of a `.csv`; `--scores`
is a one-column CSV. Pass `--features features.csv` to any command to skip encoding.

## How it works

Each dimension is a cosine similarity between L2-normalised embeddings (higher = better);
`naturalness` defaults to a *reference-free* signal (similarity to a learned centroid). The
combiner is the only learned part: `lens-eval` validates your inputs, infers the target type and
link, cross-validates each candidate combiner, applies the 1-SE rule to favour the simplest
adequate model, refits it on all your data, and computes diagnostics, all inside `fit()`.

## Requirements

- Python ≥ 3.9
- The `encoders` extra to featurize raw text
- To train: at least 50 labelled rows (fewer than 200 restricts selection to the linear
  combiner). Reference-mode dimensions need references at both fit and score time.

## Documentation

- **[Full documentation](DOCS.md)** — every argument, task mode, the combiner ladder, encoder
  configuration, the multi-reference dialogue features, persistence, and the CLI.
- **[`examples/rag_rerank.py`](examples/rag_rerank.py)** — the runnable RAG demo above.

## License

[Apache-2.0](LICENSE)

# lens-eval,  Documentation

This document is the
full reference; for a 30-second tour see the [README](README.md) and
[`examples/rag_rerank.py`](examples/rag_rerank.py).

- [Pretrained metrics on the Hub](#pretrained-metrics-on-the-hub)

- [Mental model](#mental-model)
- [The four dimensions](#the-four-dimensions)
- [Installation & optional dependencies](#installation--optional-dependencies)
- [`LENS.fit`](#lensfit)
  - [Task modes](#task-modes)
  - [Target types & link functions](#target-types--link-functions)
  - [Model selection & the combiner ladder](#model-selection--the-combiner-ladder)
- [Scoring: `score` / `compare` / `rank`](#scoring-score--compare--rank)
- [Interpretability & reports](#interpretability--reports)
- [Encoders & configuration](#encoders--configuration)
- [Bring your own features](#bring-your-own-features)
- [Persistence & the Hugging Face Hub](#persistence--the-hugging-face-hub)
- [Command-line interface](#command-line-interface)
- [Fitted attributes](#fitted-attributes)
- [Errors](#errors)
- [Limitations & gotchas](#limitations--gotchas)

---

## Pretrained metrics on the Hub

Several fitted combiners are published and loadable with `LENS.load("owner/repo")`.
They fall into two groups by how they featurise text.

**Standard-dimension metrics,  score directly from text.** These were fit on the
four standard dimensions scored against a single reference, so the stock
encoders reproduce their features:

| Repo | Task | Usage |
|---|---|---|
| `foudil/lens_focus` | RAG / best-answer selection (RewardBench-2 *Focus*) | `LENS.load("foudil/lens_focus").score(answers, references=[prompt]*n)` |
| `foudil/lens_wmt_da` | machine-translation quality (WMT direct assessment) | `LENS.load("foudil/lens_wmt_da").score(translations, references=refs)` |

For `lens_focus` the "reference" is the **prompt**,  it measures how well a
response fits the prompt, which is exactly the RAG re-ranking decision. Scores
are on the combiner's target scale (a 0–1 best-answer probability for `focus`).

**Multi-reference dialogue facets,  assemble features yourself.** The
TopicalChat-USR facet metrics (`lens_uses_knowledge`, `lens_engaging`,
`lens_maintains_context`, `lens_understandable`, `lens_natural`) were fit on a
**7-dimension** matrix built from the *same* encoders applied against *different*
references (the dialogue context, the retrieved fact, and a gold response). You
reproduce those columns with the per-axis functions and a NLI cross-encoder,
then score via `features=`:

```python
import numpy as np
from scipy.special import softmax
from sentence_transformers import CrossEncoder
from lens_eval import LENS
from lens_eval.encoders import naturalness_score, emotion_score, semantic_score

nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")
def entail(pairs):  # P(entailment); label order [contradiction, entailment, neutral]
    return softmax(nli.predict(pairs), axis=1)[:, 1]

def usr_features(responses, context, fact, gold):
    n = len(responses)
    ctx, fct, gld = [context]*n, [fact]*n, [gold]*n
    return np.column_stack([
        naturalness_score(responses),            # sim_naturalness  (reference-free)
        emotion_score(responses, gld),           # sim_emotion      ← gold response
        semantic_score(responses, ctx),          # sim_semantic_dialog ← dialogue context
        semantic_score(responses, fct),          # sim_semantic_fact   ← retrieved fact
        semantic_score(responses, gld),          # sim_semantic_ref    ← gold response
        entail(list(zip(ctx, responses))),       # sim_nli_fwd_dialog  context ⇒ reply
        entail(list(zip(responses, fct))),       # sim_nli_rev_fact    reply ⇒ fact
    ]).astype(float)

X = usr_features(responses, context, fact, gold)
score = LENS.load("foudil/lens_uses_knowledge").score(features=X)
```

The column order must match the metric's `dimensions_used_`
(`sim_naturalness, sim_emotion, sim_semantic_dialog, sim_semantic_fact,
sim_semantic_ref, sim_nli_fwd_dialog, sim_nli_rev_fact`).

---

## Mental model

A `LENS` is two stages:

1. **Featurise**,  each candidate text is reduced to a small vector of
   dimension scores (cosine similarities in `[-1, 1]`, higher = better). This
   stage uses transformer encoders and is *fixed*,  nothing is learned here.
2. **Combine**,  a *combiner* maps that vector to one quality score. This is the
   only learned part. `fit()` cross-validates a ladder of combiners and keeps
   the simplest one that fits your judgments.

You can use only stage 2 by passing a precomputed `features` matrix (no
encoders, no model downloads),  see [Bring your own features](#bring-your-own-features).

Everything happens inside `fit()`: validate inputs → infer target type & link →
featurise → capacity-gate the candidate combiners → cross-validate each → pick
the winner with the 1-SE rule → refit on all data → compute diagnostics →
populate `selection_report_`, `diagnostics_`, and `combiner_`.

---

## The four dimensions

`DIMENSIONS = ("semantic", "nli", "naturalness", "emotion")`

| Dimension | What it measures | Default encoder | Needs a reference? |
|---|---|---|---|
| `semantic` | meaning overlap with the reference | `sentence-transformers/all-mpnet-base-v2` | yes |
| `nli` | entailment-flavoured similarity | `cross-encoder/nli-deberta-v3-base` | yes |
| `naturalness` | fluency / human-likeness | `foudil/lens-naturalness-encoder` | **optional** (centroid mode) |
| `emotion` | affective alignment | `foudil/lens-emotion-encoder` | yes |

Every score is a cosine similarity between L2-normalised embeddings, clipped to
`[-1, 1]`. `naturalness` is special: by default it runs in **centroid mode**, a
reference-free signal (cosine to a learned centroid of natural text), so it
contributes even when you score without references. See
[Encoders & configuration](#encoders--configuration) for the modes.

A reference-mode dimension with no reference produces a NaN column, which
`fit()` drops automatically (with a warning).

---

## Installation & optional dependencies

The core install is light: `numpy`, `scipy`, `scikit-learn`, `huggingface_hub`.
It can fit/score from a feature matrix and run the GLM combiners, but it cannot
encode raw text. Extras add capabilities:

| Extra | Adds | Unlocks |
|---|---|---|
| _(core)_ |,  | GLM / GLM+interactions combiners, feature-matrix workflow |
| `encoders` | torch, transformers, sentence-transformers | scoring raw text |
| `ebm` | interpret-ml | the EBM (GA²M) combiner |
| `gbm` | lightgbm | the monotonic GBM combiner¹ |
| `statsmodels` | statsmodels | richer GLM inference |
| `report` | jinja2 | `report_html()` |
| `all` | all of the above | everything |

```bash
pip install lens-eval
pip install 'lens-eval[encoders]'
pip install 'lens-eval[all]'
```

¹ On macOS `lightgbm` also needs OpenMP: `brew install libomp`.

If a combiner's backend is missing, `lens-eval` quietly drops it from the
candidate set and notes it in the report (with the exact `pip install` line). It
only errors (`CombinerBackendMissing`) if *no* candidate is installable.

---

## `LENS.fit`

```python
LENS(random_state=42).fit(
    texts=None, references=None, features=None,
    scores=None, pairs=None, ranks=None, groups=None,
    primary_metric="auto", task="auto", target_type="auto", selection="auto",
    dimensions=None, hypothesized_interactions=None,
    cv_splits=5, n_range=None, verbose=False,
) -> LENS
```

You must supply the **inputs** (`texts` [+ `references`], or `features`) and
exactly one **target channel** (`scores`, `pairs`, or `ranks`+`groups`).


| Argument | Meaning |
|---|---|
| `texts` | candidate texts to encode |
| `references` | references for reference-mode dimensions |
| `features` | precomputed `(N, D)` array or DataFrame,  skips encoding |
| `scores` | scalar targets → **regression** |
| `pairs` | `(winner_idx, loser_idx)` rows → **pairwise** |
| `ranks` / `groups` | integer ranks + group ids → **ranking** (both required) |
| `primary_metric` | `"auto"` picks per task; override to pin a CV metric |
| `task` | `"auto"` (from the target channel) or force a mode |
| `target_type` | `"auto"` / `bounded` / `ordinal` / `binary` / `continuous` |
| `selection` | `"auto"` / `"fast"` / `"exhaustive"` / a combiner name |
| `dimensions` | subset/order of dimension names to use |
| `hypothesized_interactions` | pairs of dimensions for `glm_interactions` |
| `cv_splits` | outer CV folds (default 5) |
| `n_range` | if set, compute per-feature marginal-impact bins for the report |
| `verbose` | print CV progress and the report at the end |

`fit()` needs **≥ 50 rows** (`InsufficientDataError` otherwise). With `< 200`
rows and `selection="auto"` it restricts to the GLM combiner and warns.

### Task modes

The task is inferred from which target channel you pass; you rarely set `task`
explicitly.

```python
# regression,  continuous / Likert / DA scores
LENS().fit(texts=hyps, references=refs, scores=y)

# pairwise,  learn from preferences (Bradley-Terry antisymmetric expansion)
LENS().fit(texts=hyps, references=refs, pairs=[(0, 3), (5, 2), ...])

# ranking,  graded relevance within groups
LENS().fit(texts=hyps, references=refs, ranks=ranks, groups=query_ids)
```

The primary CV metric per task: **regression → Spearman**, **pairwise → AUC**,
**ranking → Kendall τ**. The winner's full panel always reports Spearman,
Kendall, Pearson, and MAE (plus AUC/Brier for binary/pairwise).

### Target types & link functions

For regression, `target_type` is auto-detected from `scores`:

| Detected | When | Link |
|---|---|---|
| `binary` | values ⊆ {0, 1} | logit |
| `bounded` | integers, or floats in `[0, 1]` / `[0, 100]` | logit |
| `continuous` | free-floating floats | identity |
| `ordinal` | (opt-in via `target_type="ordinal"`) | cumulative logit |

For **bounded** regression, `fit()` caches the target's `(min, max)`, fits in a
rescaled `[0, 1]` space, and `score()` maps predictions back to your original
range,  so you get scores on the scale you trained on.

### Model selection & the combiner ladder

Candidates are gated by sample size (`selection="auto"`), simplest first:

| Rows `n` | Candidates considered |
|---|---|
| `< 200` | `glm` |
| `200–999` | `glm`, `glm_interactions` |
| `1000–4999` | `glm`, `glm_interactions`, `ebm` |
| `≥ 5000` | `glm`, `glm_interactions`, `ebm`, `gbm` |

| Combiner | What it is | Interpretability |
|---|---|---|
| `glm` | regularised generalised linear model | signed per-dimension coefficients |
| `glm_interactions` | GLM + selected pairwise terms | main-effect coefficients + interactions |
| `ebm` | Explainable Boosting Machine (GA²M) | per-feature shape functions / importances |
| `gbm` | monotonic gradient-boosted trees | non-negative feature importances |

Other `selection` values: `"fast"` (GLM only), `"exhaustive"` (all four
regardless of `n`), or a specific name (`"glm"`, `"ebm"`, …) to force one.

Each candidate is scored by K-fold CV (GroupKFold when `groups` are present,
else StratifiedKFold for discrete targets, else KFold). The winner is chosen by
the **1-SE rule**: take the best mean, then pick the *simplest* combiner whose
mean is within one standard error of it. This biases toward interpretable,
overfit-resistant models,  in the example, `glm_interactions` had a marginally
higher mean but `glm` was within 1 SE and won.

---

## Scoring: `score` / `compare` / `rank`

```python
lens.score(texts, references=None, *, features=None, discretize=False) -> np.ndarray
lens.compare(texts_a, texts_b, references=None, *, features_a=None, features_b=None) -> np.ndarray
lens.rank(candidates, references=None, *, features=None) -> np.ndarray
lens.contributions(texts, references=None, *, features=None) -> np.ndarray  # (N, D)
```

- **`score`** returns one quality score per row, on your original target scale.
  `discretize=True` rounds (and clips to the trained range for bounded targets)
 ,  handy for Likert/DA outputs.
- **`compare`** returns `score(a) - score(b)`; positive means `a` is better.
- **`rank`** returns indices best-first (`argsort(-score)`).
- **`contributions`** returns an `(N, D)` attribution matrix (linear combiners:
  `X * coef`; EBM/GBM: SHAP-style per-feature contributions).

If the model was fit *with* references, you must pass references at score time
too, or you get a `ReferenceModeError` (raised before any encoding, so the
message is actionable).

---

## Interpretability & reports

```python
lens.report()                      # pretty text report to stdout
lens.report_html("report.html")    # standalone HTML (needs the `report` extra)
names, values, kind = lens.feature_importance()
```

`feature_importance()` returns one value per base dimension plus a `kind`:
`"coef"` (signed, GLM family) or `"importance"` (non-negative, EBM/GBM).

The report (and the `selection_report_` dict behind it) contains:

- **Setup**: task, target type, link, sample count, dimensions used.
- **Capacity gate**: candidates considered and any dropped backends (with
  install hints).
- **Candidate scores**: per-combiner CV mean ± std of the primary metric.
- **Selection**: the winner and the 1-SE reasoning.
- **Winner panel**: full metric panel on aggregated out-of-sample predictions.
- **Feature ablation**: drop-column importance,  for each dimension, the primary
  metric after replacing that column with its mean, and the delta from baseline
  (a near-zero delta means the dimension carries no signal).
- **Fitted combiner**: intercept + coefficients (or feature importances).
- **Diagnostics**: residual summary, calibration (ECE/Brier), feature
  correlation, monotonicity check, outlier flags.

Pass `n_range=k` to `fit()` to also compute per-feature marginal-impact bins
(mean contribution within `k` quantile bins of each dimension).

---

## Encoders & configuration

The encoder layer is configured per process via `configure()`:

```python
from lens_eval import configure
configure(
    device="cuda",                 # or "mps" / "cpu"; auto-detected if unset
    batch_size=64,
    paths={"semantic": "my-org/my-encoder"},   # override any dimension's encoder
    naturalness_mode="centroid",   # "centroid" (default) or "reference"
    naturalness_centroid=vec,      # supply your own centroid vector
)
```

`configure()` is idempotent and takes a *partial* dict,  only the keys you pass
are touched. Changing `paths` / `device` / `batch_size` invalidates the lazy
encoder cache. `free()` drops cached encoders to release GPU memory.

**Naturalness modes.** In `centroid` mode (default), naturalness is a
reference-free score: cosine to a centroid of natural text. `lens-eval` ships a
centroid for its bundled NatureEncoder and auto-loads it. If that encoder isn't
reachable in your environment, it falls back to `reference` mode with a warning;
pass your own `naturalness_centroid` or `paths={"naturalness": ...}` to keep
centroid mode. In `reference` mode, naturalness behaves like the other
dimensions (cosine to the reference).

**Direct dimension functions** (each returns an `np.ndarray`):

```python
from lens_eval import semantic_score, nli_score, naturalness_score, emotion_score, featurize
semantic_score(texts, refs)
naturalness_score(texts)               # references optional in centroid mode
X = featurize(texts, references=refs, dimensions=("semantic", "nli"))  # (N, D)
```

---

## Bring your own features

If your dimension scores already exist (e.g. in a CSV), skip encoding entirely:

```python
import numpy as np
X = np.column_stack([semantic, nli, naturalness, emotion])  # (N, 4)
lens = LENS().fit(features=X, scores=y)
preds = lens.score(features=X_new)
```

`features` may be a NumPy array (matched positionally to `dimensions`, defaulting
to the four standard names) or a **pandas DataFrame** (matched by column name, 
at fit time the headers *become* the dimension names; at score time the fitted
dimensions are selected by name, so column order and extra columns don't
matter). A column that is entirely NaN is dropped with a warning.

This is also the fastest path for examples and tests,  no model downloads, fully
deterministic.

---

## Persistence & the Hugging Face Hub

```python
lens.save("./my-lens")
lens = LENS.load("./my-lens")
```

A saved model directory contains:

```
my-lens/
  manifest.json             # version, encoder fingerprints, fit metadata
  combiner.pkl              # the fitted combiner
  selection_report.json     # full structured report (CV scores, diagnostics)
  naturalness_centroid.npy  # only when centroid mode was active
```

Encoders are **not** saved,  the manifest records their paths and a fingerprint
of any local checkpoints. `load()` re-points the encoder config at them. If both
the saved and runtime sides fingerprint a local encoder and they differ, `load()`
raises `EncoderVersionMismatchError` rather than silently miscalibrate.

**Hugging Face Hub.** `load()` accepts an `owner/repo` id (an existing local path
always wins over a same-named repo). Push a fitted model with:

```python
url = lens.push_to_hub("owner/my-lens", private=False)
lens = LENS.load("owner/my-lens", revision="v1")   # hub kwargs forwarded
```

Both forward extra kwargs (`token`, `revision`, `cache_dir`, …) to the
underlying `huggingface_hub` calls.

---

## Command-line interface

```bash
lens-eval fit   --texts hyp.txt --refs ref.txt --scores scores.csv --output ./my-lens [--html report.html]
lens-eval score --model ./my-lens --texts new.txt --refs ref.txt --output preds.csv
lens-eval report ./my-lens [--html report.html]
```

`fit` flags mirror the API: `--task`, `--target-type`, `--selection`,
`--naturalness-mode`, `--verbose`, and the target channels `--scores` /
`--pairs` / `--ranks` + `--groups`. Inputs: `--texts`/`--refs` take one item per
line (`.txt`) or the first column of a `.csv`; `--scores` is a one-column CSV;
`--pairs` is a two-column CSV; `--ranks`/`--groups` are one-column integer CSVs.

Pass `--features features.csv` to any command to skip encoding. The feature CSV
has one column per dimension; column order follows `--dimensions` for `fit`, or
the model's `dimensions_used_` for `score`.

---

## Fitted attributes

After `fit()`, a `LENS` exposes sklearn-style trailing-underscore attributes:

| Attribute | Meaning |
|---|---|
| `combiner_` | the fitted combiner object |
| `combiner_type_` | winning combiner name |
| `task_` | `regression` / `pairwise` / `ranking` |
| `target_type_` | `bounded` / `ordinal` / `binary` / `continuous` |
| `link_function_` | `identity` / `logit` / `cumulative_logit` |
| `dimensions_used_` | dimensions actually fitted (after dropping empty columns) |
| `target_range_` | cached `(min, max)` for bounded rescaling, else `None` |
| `cv_scores_` | per-combiner CV mean/std/per-fold |
| `selection_report_` | the full structured report dict |
| `diagnostics_` | the diagnostics dict |

---

## Errors

All inherit from `LensEvalError`, so `except LensEvalError` catches any of them:

| Exception | Raised when |
|---|---|
| `InsufficientDataError` | fewer than 50 rows |
| `DegenerateTargetError` | the target has zero variance |
| `AmbiguousTaskError` | more than one target channel passed |
| `ReferenceModeError` | score-time reference mode disagrees with training |
| `EncoderVersionMismatchError` | `load()` finds mismatched encoder fingerprints |
| `CombinerBackendMissing` | a requested combiner's backend isn't installed |

---

## Limitations & gotchas

- **≥ 50 rows to fit**, and `< 200` rows restricts auto-selection to the linear
  combiner. Small data → simple model, by design.
- **Reference mode is sticky.** Fit with references ⇒ score with references.
- **Encoding needs the `encoders` extra.** Core install is feature-matrix only.
- **The four dimensions are correlated** (semantic and nli especially); the
  report's feature-correlation diagnostic surfaces this. Coefficients are still
  interpretable but read them as a set, not in isolation.
- **`naturalness` centroid is tied to its encoder's vector space.** Pointing
  `paths["naturalness"]` at a different encoder without a matching centroid will
  error or fall back to reference mode,  supply a matching `naturalness_centroid`.

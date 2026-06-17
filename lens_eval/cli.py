"""`lens-eval` CLI — thin wrapper around the Python API.

    lens-eval fit \
        --texts hyp.txt --refs ref.txt --scores scores.csv \
        --output ./my-lens --task auto --verbose

    lens-eval score \
        --model ./my-lens \
        --texts new_hyp.txt --refs new_ref.txt \
        --output predictions.csv

    lens-eval report ./my-lens --html report.html

All three accept ``--features path.csv`` to skip encoding entirely. The
features CSV must have one column per dimension; column order follows the
LENS ``dimensions_used_`` for ``score``, or the user-passed ``--dimensions``
for ``fit``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


def _read_lines(path: str | Path) -> List[str]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".csv":
        return [row[0] for row in csv.reader(text.splitlines()) if row]
    return [ln for ln in text.splitlines() if ln]


def _read_scalar_csv(path: str | Path) -> np.ndarray:
    rows: List[float] = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                rows.append(float(row[0]))
            except ValueError:
                # Header row — skip silently.
                continue
    return np.asarray(rows, dtype=float)


def _read_features_csv(path: str | Path) -> np.ndarray:
    arr = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                arr.append([float(x) for x in row])
            except ValueError:
                continue
    return np.asarray(arr, dtype=float)


def _read_pairs_csv(path: str | Path) -> np.ndarray:
    pairs = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                a, b = int(row[0]), int(row[1])
            except (ValueError, IndexError):
                continue
            pairs.append((a, b))
    if not pairs:
        raise ValueError(f"no pair rows parsed from {path}")
    return np.asarray(pairs, dtype=int)


def _read_int_column_csv(path: str | Path) -> np.ndarray:
    vals: List[int] = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                vals.append(int(row[0]))
            except ValueError:
                continue
    if not vals:
        raise ValueError(f"no integer rows parsed from {path}")
    return np.asarray(vals, dtype=int)


def _cmd_fit(args) -> int:
    from . import encoders as enc
    from ._validate import validate_task_channels
    from .lens import LENS

    if args.naturalness_mode:
        enc.configure(naturalness_mode=args.naturalness_mode)

    texts  = _read_lines(args.texts)           if args.texts    else None
    refs   = _read_lines(args.refs)            if args.refs     else None
    feats  = _read_features_csv(args.features) if args.features else None
    y      = _read_scalar_csv(args.scores)     if args.scores   else None
    pairs  = _read_pairs_csv(args.pairs)       if args.pairs    else None
    ranks  = _read_int_column_csv(args.ranks)  if args.ranks    else None
    groups = _read_int_column_csv(args.groups) if args.groups   else None

    try:
        channel, _ = validate_task_channels(
            scores=y, pairs=pairs, ranks=ranks, groups=groups,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    lens = LENS()
    lens.fit(
        texts=texts,
        references=refs,
        features=feats,
        scores=y, pairs=pairs, ranks=ranks, groups=groups,
        task=args.task,
        target_type=args.target_type,
        selection=args.selection,
        verbose=args.verbose,
    )
    lens.save(args.output)
    if args.html:
        lens.report_html(args.html)
    print(f"saved fitted LENS to {args.output} ({channel} channel)")
    return 0


def _cmd_score(args) -> int:
    from .lens import LENS

    lens = LENS.load(args.model)
    texts = _read_lines(args.texts)            if args.texts    else None
    refs  = _read_lines(args.refs)             if args.refs     else None
    feats = _read_features_csv(args.features)  if args.features else None

    if texts is None and feats is None:
        print("error: provide --texts or --features", file=sys.stderr)
        return 2

    scores = lens.score(texts, references=refs, features=feats)

    out = Path(args.output)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score"])
        for s in scores:
            w.writerow([float(s)])
    print(f"wrote {len(scores)} scores to {out}")
    return 0


def _cmd_report(args) -> int:
    from .lens import LENS

    lens = LENS.load(args.model)
    lens.report()
    if args.html:
        lens.report_html(args.html)
        print(f"wrote HTML report to {args.html}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="lens-eval", description="lens-eval CLI")
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("fit", help="fit a LENS combiner")
    pf.add_argument("--texts",    help="path to candidate texts (txt or csv)")
    pf.add_argument("--refs",     help="path to references (txt or csv)")
    pf.add_argument("--features", help="path to precomputed feature matrix CSV (D columns)")
    pf.add_argument("--scores",   help="scalar targets CSV (1 column)")
    pf.add_argument("--pairs",    help="pairwise CSV: 2 columns (winner_idx, loser_idx)")
    pf.add_argument("--ranks",    help="ranking CSV: 1 column of integer ranks")
    pf.add_argument("--groups",   help="ranking grouping CSV: 1 column of group ids — REQUIRED with --ranks")
    pf.add_argument("--output",   required=True, help="where to save the fitted LENS directory")
    pf.add_argument("--task",        default="auto", choices=("auto", "regression", "pairwise", "ranking"))
    pf.add_argument("--target-type", default="auto",
                    choices=("auto", "bounded", "ordinal", "binary", "continuous"))
    pf.add_argument("--selection",   default="auto",
                    choices=("auto", "fast", "exhaustive", "glm", "glm_interactions", "ebm", "gbm"))
    pf.add_argument("--naturalness-mode", default="centroid", choices=("centroid", "reference"))
    pf.add_argument("--html",    help="optional: also write an HTML report to this path")
    pf.add_argument("--verbose", action="store_true")
    pf.set_defaults(func=_cmd_fit)

    ps = sub.add_parser("score", help="score new texts with a saved LENS")
    ps.add_argument("--model",    required=True, help="path to a saved LENS directory")
    ps.add_argument("--texts",    help="path to candidate texts (txt or csv)")
    ps.add_argument("--refs",     help="path to references (txt or csv)")
    ps.add_argument("--features", help="path to precomputed feature matrix CSV (D columns)")
    ps.add_argument("--output",   required=True, help="output CSV path for predictions")
    ps.set_defaults(func=_cmd_score)

    pr = sub.add_parser("report", help="print/render the selection report from a saved LENS")
    pr.add_argument("model", help="path to a saved LENS directory")
    pr.add_argument("--html", help="optional: write an HTML report to this path too")
    pr.set_defaults(func=_cmd_report)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

"""Selection-report rendering.

Two output paths:
- `format_text(report)` — pretty plain text, matches the spec §6.1 example.
- `format_html(report, diagnostics, ...)` — optional HTML write.

The report dict shape is fixed (see LENS.selection_report_) so callers can
also consume it programmatically without parsing the formatted text.
"""

from __future__ import annotations

import platform
from typing import Any, Dict, List, Optional


# Per-combiner description + install hint. Surfaced in the "dropped (backend
# not installed)" section of the report so users see *exactly* what to install.
_DROPPED_HINTS: Dict[str, Dict[str, str]] = {
    "ebm": {
        "what":    "Explainable Boosting Machine (GA²M).",
        "needs":   "interpret-ml",
        "install": "pip install 'lens-eval[ebm]'",
        "extra":   "",
    },
    "gbm": {
        "what":    "monotonic gradient-boosted trees.",
        "needs":   "lightgbm",
        "install": "pip install 'lens-eval[gbm]'",
        # macOS users typically also need libomp for lightgbm to load.
        "extra":   "brew install libomp" if platform.system() == "Darwin" else "",
    },
}


def _isnan(v) -> bool:
    return isinstance(v, float) and v != v


def _render_dropped(dropped: List[str]) -> List[str]:
    """Multi-line rendering of dropped backends with install hints."""
    if not dropped:
        return []
    # Indentation here matches the rest of the report layout (21-char prefix)
    # so the dropped section visually aligns under "Capacity gate".
    lines = [" " * 21 + "dropped (backend not installed):"]
    for name in dropped:
        hint = _DROPPED_HINTS.get(name)
        if hint is None:
            lines.append(f"{' ' * 23}• {name}")
            continue
        lines.append(f"{' ' * 23}• {name} — {hint['what']} (needs {hint['needs']})")
        install_line = f"{' ' * 25}{hint['install']}"
        # Only add the macOS-only extra when present (set in _DROPPED_HINTS).
        if hint.get("extra"):
            install_line += f"   (macOS: {hint['extra']})"
        lines.append(install_line)
    lines.append(f"{' ' * 23}install everything with: pip install 'lens-eval[all]'")
    return lines


def format_text(report: Dict[str, Any]) -> str:
    """Render a selection report to plain text. Stable line shape; for humans."""
    # 75-char width matches the spec example and fits in a standard 80-col
    # terminal even with a small left margin. `bar` is reused as a separator.
    L = 75
    out: List[str] = []
    bar = "─" * L

    # --- header block: task / target / link / sample count -------------
    out.append(bar)
    out.append(" LENS combiner selection report")
    out.append(bar)
    out.append(f" Task              : {report['task']}")
    out.append(f" Target type       : {report['target_type']}")
    out.append(f" Link function     : {report['link']}")
    out.append(f" Samples (train)   : {report['n_samples']:,}")
    out.append(f" Dimensions used   : {', '.join(report['dimensions_used'])}")
    out.append("")
    # --- candidate set + CV scheme -------------------------------------
    cand = report.get("candidates", [])
    dropped = report.get("dropped_backends", [])
    out.append(f" Capacity gate     : n={report['n_samples']:,} → candidates: "
               f"{', '.join(cand) if cand else '(none)'}")
    out.extend(_render_dropped(dropped))
    out.append(f" Selection metric  : {report['primary_metric']} (primary)")
    out.append(f" CV scheme         : {report['cv_splits']}-fold outer, seed={report['random_state']}")
    out.append("")
    # --- candidate scores table ----------------------------------------
    primary_name = report.get("primary_metric", "primary")
    out.append(bar)
    out.append(f" Candidate scores — {primary_name} (mean ± std across outer folds)")
    out.append(bar)
    out.append(f" {'Combiner':<24} {primary_name + ' mean ± std':>22}")
    out.append(" " + "─" * (L - 1))
    for r in report["cv_scores"]:
        mean = r.get("primary_mean", float("nan"))
        std  = r.get("primary_std",  float("nan"))
        out.append(f" {r['combiner_type']:<24} {mean:>+10.4f} ± {std:.4f}")
    out.append("")
    out.append(bar)
    out.append(" Selection")
    out.append(bar)
    out.append(f" Winner: {report['winner']}")
    out.append(" Reason: " + report.get("reason", ""))
    out.append("")

    # --- winner panel: full metric set on aggregated OOF predictions ----
    panel = report.get("winner_panel") or {}
    if panel:
        out.append(bar)
        out.append(f" Winner panel — full metrics on aggregated out-of-sample predictions")
        out.append(bar)
        for k in ("spearman", "kendall", "pearson", "mae", "auc", "brier"):
            if k in panel:
                v = panel[k]
                out.append(f"   {k:<10} {v:+.4f}" if not _isnan(v) else f"   {k:<10} (n/a)")
        out.append("")

    # --- feature ablation --------------------------------------------
    abl = report.get("feature_ablation") or {}
    if abl.get("by_feature"):
        primary_name = report.get("primary_metric", "primary")
        baseline = abl.get("baseline", float("nan"))
        out.append(bar)
        out.append(f" Feature ablation — drop-column on training data ({primary_name})")
        out.append(bar)
        out.append(f"   baseline   {baseline:+.4f}")
        out.append(f" {'feature':<24} {'masked':>10} {'Δ from baseline':>18}")
        out.append(" " + "─" * (L - 1))
        for row in abl["by_feature"]:
            out.append(
                f" {row['name']:<24} {row['score']:>+10.4f} {row['delta']:>+18.4f}"
            )
        out.append("")

    # --- marginal impact bins (opt-in via n_range) ---------------------
    marg = report.get("marginal_impact")
    if marg:
        out.append(bar)
        out.append(" Marginal impact by quantile bin (mean per-row contribution)")
        out.append(bar)
        for name, bins in marg.items():
            out.append(f" {name}")
            out.append(f"   {'bin':<5} {'x range':<22} {'x_mean':>10} "
                       f"{'contribution_mean':>20} {'n':>7}")
            for k, b in enumerate(bins):
                rng = f"[{b['x_lo']:+.3f}, {b['x_hi']:+.3f}]"
                out.append(
                    f"   {k:<5} {rng:<22} {b['x_mean']:>+10.4f} "
                    f"{b['contribution_mean']:>+20.4f} {b['n']:>7}"
                )
            out.append("")

    # --- fitted coefficients / importances ----------------------------
    # Two distinct shapes here: GLM-family tiers expose `coef` (signed
    # linear weights), tree-family tiers expose `feature_importances`
    # (non-negative). We render whichever is present.
    coef = report.get("fitted_coefficients")
    if coef:
        out.append(bar)
        out.append(f" Fitted combiner: {report['winner']}")
        out.append(bar)
        intercept = coef.get("intercept")
        if intercept is not None:
            out.append(f" intercept = {intercept:+.4f}")
        c = coef.get("coef")
        # Fall back to dimension names from the report when the combiner
        # didn't attach its own — keeps the table readable in all cases.
        names = coef.get("feature_names") or report.get("dimensions_used")
        if c is not None:
            for i, v in enumerate(c):
                name = names[i] if names and i < len(names) else f"x{i}"
                out.append(f"   {name:<28} {v:+.4f}")
        importances = coef.get("feature_importances")
        if importances:
            out.append(" Feature importances:")
            for i, imp in enumerate(importances):
                name = names[i] if names and i < len(names) else f"x{i}"
                # EBM packs raw + interaction importances mixed together;
                # the dict entries can be lists, so we filter to scalars.
                if isinstance(imp, (int, float)):
                    out.append(f"   {name:<28} {imp:+.4f}")
        out.append("")

    diag = report.get("diagnostics_summary")
    if diag:
        out.append(bar)
        out.append(" Diagnostics")
        out.append(bar)
        for k, v in diag.items():
            out.append(f"   {k:<22}: {v}")
        out.append("")

    warnings = report.get("warnings", [])
    if warnings:
        out.append(bar)
        out.append(" Warnings")
        out.append(bar)
        for w in warnings:
            out.append(f"   • {w}")
        out.append("")

    out.append(bar)
    return "\n".join(out)


def summarise_diagnostics(diag: Dict[str, Any]) -> Dict[str, str]:
    """Compress the full diagnostics dict into one-liners per section."""
    if not diag:
        return {}
    # One key per diagnostic section — each value is a short string the
    # report layer prints as-is. Keeps the report scannable instead of
    # dumping the full nested dict.
    summary: Dict[str, str] = {}
    r = diag.get("residuals", {})
    if r:
        summary["Residuals"] = (
            f"mean={r['mean']:+.3f}  std={r['std']:.3f}  "
            f"|max|={r['abs_max']:.3f}"
        )
    cal = diag.get("calibration")
    if cal:
        summary["Calibration"] = f"ECE={cal['ece']:.3f}  Brier={cal['brier']:.3f}"
    fc = diag.get("feature_correlations")
    if fc:
        summary["Feature correlations"] = f"max |off-diagonal| = {fc['max_offdiag']:.3f}"
    mono = diag.get("monotonicity", {})
    if mono:
        non_mono = [n for n, entry in mono.items()
                    if isinstance(entry, dict) and entry.get("status") == "non-monotone"]
        if non_mono:
            summary["Monotonicity"] = f"non-monotone: {', '.join(non_mono)}"
        else:
            summary["Monotonicity"] = "all dimensions monotone"
    out = diag.get("outliers", {})
    if out:
        summary["Outliers"] = f"|std-resid| > 3  →  {out['count']} rows flagged"
    return summary


def format_html(report: Dict[str, Any], diagnostics: Optional[Dict[str, Any]] = None) -> str:
    """Minimal-but-real HTML output. Uses jinja2 if available, else plain string."""
    try:
        from jinja2 import Template
    except ImportError:
        # jinja2 is an optional dep. Without it we still produce a valid HTML
        # file by wrapping the text report in <pre>. HTML-escape < and > so
        # the rare report containing them doesn't break the page.
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>LENS selection report</title></head><body><pre>"
            + format_text(report).replace("<", "&lt;").replace(">", "&gt;")
            + "</pre></body></html>"
        )

    tmpl = Template(_HTML_TEMPLATE)
    # Pre-flatten the dropped backends so the template doesn't need to call
    # _DROPPED_HINTS.get itself — keeps the template Jinja-only.
    dropped_info = [
        {
            "name": d,
            **(_DROPPED_HINTS.get(d, {"what": "", "needs": "", "install": "", "extra": ""})),
        }
        for d in report.get("dropped_backends", [])
    ]
    return tmpl.render(report=report, diagnostics=diagnostics or {}, dropped_info=dropped_info)


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>LENS selection report</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; max-width: 880px; margin: 2em auto; color: #222; }
h1, h2 { font-weight: 500; }
table { border-collapse: collapse; margin: 1em 0; width: 100%; }
th, td { padding: 0.4em 0.7em; text-align: left; border-bottom: 1px solid #eee; }
th { background: #fafafa; }
.muted { color: #888; }
.winner { font-weight: 600; }
.warn { color: #b85c00; }
pre { background: #f7f7f7; padding: 0.8em; border-radius: 6px; }
</style>
</head><body>
  <h1>LENS combiner selection report</h1>
  <table>
    <tr><th>Task</th><td>{{ report.task }}</td></tr>
    <tr><th>Target type</th><td>{{ report.target_type }}</td></tr>
    <tr><th>Link</th><td>{{ report.link }}</td></tr>
    <tr><th>Samples</th><td>{{ report.n_samples }}</td></tr>
    <tr><th>Dimensions</th><td>{{ report.dimensions_used | join(", ") }}</td></tr>
    <tr><th>Candidates</th><td>{{ report.candidates | join(", ") }}</td></tr>
    {% if dropped_info %}
    <tr><th>Dropped backends</th><td>
      <ul style="margin:0; padding-left:1.1em;">
      {% for d in dropped_info %}
        <li>
          <strong>{{ d.name }}</strong>{% if d.what %} — {{ d.what }}{% endif %}
          {% if d.needs %}<span class="muted">(needs {{ d.needs }})</span>{% endif %}<br>
          {% if d.install %}<code>{{ d.install }}</code>{% endif %}
          {% if d.extra %} <span class="muted">— also: <code>{{ d.extra }}</code></span>{% endif %}
        </li>
      {% endfor %}
      </ul>
      <span class="muted">install all optional backends: <code>pip install 'lens-eval[all]'</code></span>
    </td></tr>
    {% endif %}
    <tr><th>Primary metric</th><td>{{ report.primary_metric }}</td></tr>
    <tr><th>Winner</th><td class="winner">{{ report.winner }}</td></tr>
    <tr><th>Reason</th><td>{{ report.reason }}</td></tr>
  </table>

  <h2>Candidate CV scores ({{ report.primary_metric }})</h2>
  <table>
    <thead><tr><th>Combiner</th><th>{{ report.primary_metric }} mean ± std</th></tr></thead>
    {% for r in report.cv_scores %}
    <tr>
      <td>{{ r.combiner_type }}</td>
      <td>{{ "%+.4f" | format(r.primary_mean) }} ± {{ "%.4f" | format(r.primary_std) }}</td>
    </tr>
    {% endfor %}
  </table>

  {% if report.winner_panel %}
  <h2>Winner panel — full metrics on aggregated out-of-sample predictions</h2>
  <table>
    {% for k, v in report.winner_panel.items() %}
    <tr><th>{{ k }}</th><td>{{ "%+.4f" | format(v) }}</td></tr>
    {% endfor %}
  </table>
  {% endif %}

  {% if report.feature_ablation and report.feature_ablation.by_feature %}
  <h2>Feature ablation ({{ report.primary_metric }})</h2>
  <p class="muted">baseline = <code>{{ "%+.4f" | format(report.feature_ablation.baseline) }}</code></p>
  <table>
    <thead><tr><th>feature</th><th>masked</th><th>Δ from baseline</th></tr></thead>
    {% for row in report.feature_ablation.by_feature %}
    <tr>
      <td>{{ row.name }}</td>
      <td>{{ "%+.4f" | format(row.score) }}</td>
      <td>{{ "%+.4f" | format(row.delta) }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  {% if report.marginal_impact %}
  <h2>Marginal impact by quantile bin</h2>
  {% for name, bins in report.marginal_impact.items() %}
  <h3>{{ name }}</h3>
  <table>
    <thead><tr><th>bin</th><th>x range</th><th>x_mean</th><th>contribution_mean</th><th>n</th></tr></thead>
    {% for b in bins %}
    <tr>
      <td>{{ loop.index0 }}</td>
      <td>[{{ "%+.3f" | format(b.x_lo) }}, {{ "%+.3f" | format(b.x_hi) }}]</td>
      <td>{{ "%+.4f" | format(b.x_mean) }}</td>
      <td>{{ "%+.4f" | format(b.contribution_mean) }}</td>
      <td>{{ b.n }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endfor %}
  {% endif %}

  {% if report.fitted_coefficients %}
  <h2>Fitted combiner coefficients</h2>
  <pre>{{ report.fitted_coefficients | tojson(indent=2) }}</pre>
  {% endif %}

  {% if report.diagnostics_summary %}
  <h2>Diagnostics</h2>
  <table>
    {% for k, v in report.diagnostics_summary.items() %}
    <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
    {% endfor %}
  </table>
  {% endif %}

  {% if report.warnings %}
  <h2 class="warn">Warnings</h2>
  <ul>
    {% for w in report.warnings %}<li class="warn">{{ w }}</li>{% endfor %}
  </ul>
  {% endif %}
</body></html>
"""

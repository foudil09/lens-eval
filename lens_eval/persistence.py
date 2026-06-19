"""Save / load a fitted LENS.

Layout:
    my-lens/
        manifest.json             # version + encoder fingerprint + fit metadata
        combiner.pkl              # the fitted combiner (any tier)
        selection_report.json     # full structured report (CV scores, diagnostics)
        naturalness_centroid.npy  # optional, only when centroid mode is active

Encoders are NOT saved — the manifest records their paths/hashes; ``load_lens``
calls :func:`lens_eval.encoders.configure` to point the module at them again.
``LENS.report_html(path)`` regenerates HTML on demand.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING
import math

import numpy as np

if TYPE_CHECKING:
    from .lens import LENS

# Hugging Face ``owner/repo`` id: exactly one slash, each segment limited to the
# chars the Hub allows. Matched only after ruling out an existing local path, so
# a real folder named ``a/b`` always wins over a same-named repo id.
_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


def _resolve_model_dir(path: str | Path, **hub_kwargs) -> Path:
    """Resolve ``path`` to a local model directory.

    An existing local path is used as-is. Otherwise, if ``path`` looks like a
    Hugging Face ``owner/repo`` id the repo is fetched with ``snapshot_download``
    and the cached directory is returned. Anything else is a missing local path.

    ``hub_kwargs`` (e.g. ``revision``, ``token``, ``cache_dir``) are forwarded to
    ``snapshot_download``.
    """
    p = Path(path)
    if p.exists():
        if not p.is_dir():
            raise NotADirectoryError(f"{p} exists but is not a directory")
        return p

    if isinstance(path, str) and _HF_REPO_ID.match(path):
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=path, repo_type="model", **hub_kwargs))

    raise FileNotFoundError(
        f"No local directory {str(path)!r}, and it is not a valid Hugging Face "
        "repo id (expected 'owner/repo')."
    )


def push_lens_to_hub(lens: "LENS", repo_id: str, *, private: bool = False,
                     commit_message: str | None = None, **upload_kwargs) -> str:
    """Save ``lens`` to a temp dir and upload it to ``repo_id`` on the Hub.

    Returns the commit URL. ``upload_kwargs`` (e.g. ``token``, ``revision``) are
    forwarded to ``upload_folder``.
    """
    import tempfile
    from huggingface_hub import create_repo, upload_folder

    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        save_lens(lens, d)
        return upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=d,
            commit_message=commit_message or "Upload LENS combiner",
            **upload_kwargs,
        )


def save_lens(lens: "LENS", path: str | Path) -> None:
    from . import __version__, encoders as enc

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    manifest = {
        "lens_eval_version": __version__,
        "encoder_manifest":  enc.manifest(),
        "fit_metadata": {
            "task":             lens.task_,
            "target_type":      lens.target_type_,
            "link_function":    lens.link_function_,
            "dimensions_used":  list(lens.dimensions_used_),
            "combiner_type":    lens.combiner_type_,
            "hyperparameters":  dict(lens.combiner_.hyperparameters)
                                if hasattr(lens.combiner_, "hyperparameters") else {},
            "n_train":          int(lens._n_train),
            "random_state":     int(lens.random_state),
            "reference_mode":   "with-reference" if lens._fitted_with_references else "reference-free",
            "target_range":     list(lens.target_range_) if lens.target_range_ is not None else None,
        },
    }
    (p / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    with open(p / "combiner.pkl", "wb") as f:
        pickle.dump(lens.combiner_, f, protocol=4)

    (p / "selection_report.json").write_text(
        json.dumps(lens.selection_report_, indent=2, default=_json_default, sort_keys=True)
    )

    centroid = enc._CONFIG["naturalness_centroid"]
    if centroid is not None:
        np.save(p / "naturalness_centroid.npy", np.asarray(centroid, dtype=np.float32))


def load_lens(path: str | Path) -> "LENS":
    from . import encoders as enc
    from .errors import EncoderVersionMismatchError
    from .lens import LENS

    p = Path(path)
    manifest = json.loads((p / "manifest.json").read_text())
    # selection_report.json only feeds reporting attrs (.report()/cv_scores_/
    # diagnostics_); score() needs only the combiner + manifest, so it's optional.
    report_path = p / "selection_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    with open(p / "combiner.pkl", "rb") as f:
        combiner = pickle.load(f)

    centroid = None
    if (p / "naturalness_centroid.npy").exists():
        centroid = np.load(p / "naturalness_centroid.npy")

    enc.configure(
        paths=manifest["encoder_manifest"]["paths"],
        naturalness_mode=manifest["encoder_manifest"].get("naturalness_mode", "centroid"),
        naturalness_centroid=centroid,
    )

    fm = manifest["fit_metadata"]
    lens = LENS(random_state=fm.get("random_state", 42))
    lens.combiner_                = combiner
    lens.combiner_type_           = fm["combiner_type"]
    lens.task_                    = fm["task"]
    lens.target_type_             = fm["target_type"]
    lens.link_function_           = fm["link_function"]
    lens.dimensions_used_         = tuple(fm.get("dimensions_used") or [])
    lens.selection_report_        = report
    lens.cv_scores_               = {
        r["combiner_type"]: {
            "mean":     r.get("primary_mean"),
            "std":      r.get("primary_std"),
            "per_fold": r.get("per_fold", []),
        }
        for r in report.get("cv_scores", [])
    }
    lens.diagnostics_             = report.get("diagnostics", {})
    lens._fitted_with_references  = fm.get("reference_mode") != "reference-free"
    lens._n_train                 = int(fm.get("n_train", 0))
    tr = fm.get("target_range")
    lens.target_range_            = tuple(tr) if tr is not None else None
    lens._fitted_                 = True

    # Refuse load if both sides fingerprinted and they differ — predictions
    # would be silently miscalibrated.
    runtime_hashes = enc.manifest()["hashes"]
    saved_hashes   = manifest["encoder_manifest"].get("hashes", {})
    mismatches = [
        f"{d}: saved={h} runtime={rh}"
        for d, h in saved_hashes.items()
        if h is not None and (rh := runtime_hashes.get(d)) is not None and h != rh
    ]
    if mismatches:
        raise EncoderVersionMismatchError(
            "Encoder hash mismatch — saved combiner was fit with different encoder weights:\n  "
            + "\n  ".join(mismatches)
        )
    return lens


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.floating) and not np.isfinite(o):
        return None
    raise TypeError(f"can't serialise {type(o)}")



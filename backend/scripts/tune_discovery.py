"""Phase 2 prep — PCMCI discovery tuning sweep on the synthetic SCM.

Measurement only: this script does NOT change any defaults in
app/discovery.py. It exists so the Phase 2 build day starts from measured
options instead of guesswork.

Sweep design (fixed seed = 0 for reproducibility):
  * dataset sizes n in {600, 2000} from app.data_loader.generate()
  * pc_alpha in {0.001, 0.005, 0.01, 0.05} at tau_max = 2
  * tau_max in {2, 3} at the best pc_alpha (per dataset size)

Every config is scored with app.discovery.edge_metrics against the known
ground truth (precision / recall / Edge F1 / SHD-style count). Results are
written to discovery_tuning_results.json next to this script, and a ranked
summary (best by Edge F1, tie-break: higher precision) is printed.

    python scripts/tune_discovery.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import data_loader
from app.discovery import edge_metrics, run_discovery

SEED = 0
SIZES = [600, 2000]
PC_ALPHAS = [0.001, 0.005, 0.01, 0.05]
TAU_MAXES = [2, 3]
RESULTS_PATH = Path(__file__).resolve().parent / "discovery_tuning_results.json"


def run_config(df, ground_truth, n_steps: int, tau_max: int, pc_alpha: float) -> dict:
    """Run one discovery config (PCMCI chatter suppressed) and score it."""
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        edges = run_discovery(df, tau_max=tau_max, pc_alpha=pc_alpha)
    runtime = time.perf_counter() - start
    metrics = edge_metrics(edges, ground_truth)
    row = {
        "n_steps": n_steps,
        "tau_max": tau_max,
        "pc_alpha": pc_alpha,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "edge_f1": metrics["edge_f1"],
        "shd_style": metrics["shd_style"],
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "runtime_s": round(runtime, 2),
    }
    print(
        f"  n={n_steps:>5}  tau_max={tau_max}  pc_alpha={pc_alpha:<6} "
        f"P={row['precision']:.4f}  R={row['recall']:.4f}  F1={row['edge_f1']:.4f}  "
        f"SHD={row['shd_style']:>2}  (TP={row['true_positives']}, "
        f"FP={row['false_positives']}, FN={row['false_negatives']})  "
        f"[{row['runtime_s']:.1f}s]"
    )
    return row


def rank_key(row: dict):
    """Best = highest Edge F1; tie-break on higher precision."""
    return (row["edge_f1"], row["precision"])


def main() -> int:
    all_rows: list[dict] = []

    for n_steps in SIZES:
        print(f"\n=== Synthetic SCM, n={n_steps}, seed={SEED} ===")
        df, ground_truth, _ = data_loader.generate(n_steps=n_steps, seed=SEED)

        alpha_rows = [
            run_config(df, ground_truth, n_steps, tau_max=2, pc_alpha=alpha)
            for alpha in PC_ALPHAS
        ]
        all_rows.extend(alpha_rows)
        best_alpha = max(alpha_rows, key=rank_key)["pc_alpha"]
        print(f"  -> best pc_alpha at tau_max=2: {best_alpha}")

        for tau_max in TAU_MAXES:
            if tau_max == 2:
                continue  # already run in the alpha sweep
            all_rows.append(
                run_config(df, ground_truth, n_steps, tau_max=tau_max,
                           pc_alpha=best_alpha)
            )

    RESULTS_PATH.write_text(json.dumps(
        {"seed": SEED, "sizes": SIZES, "pc_alphas": PC_ALPHAS,
         "tau_maxes": TAU_MAXES, "results": all_rows},
        indent=2,
    ))
    print(f"\nResults written to {RESULTS_PATH}")

    print("\n=== Ranked summary (all configs, by Edge F1, tie-break precision) ===")
    print(f"{'n':>6} {'tau':>4} {'alpha':>7} {'P':>8} {'R':>8} {'F1':>8} {'SHD':>4} {'time':>8}")
    for row in sorted(all_rows, key=rank_key, reverse=True):
        print(
            f"{row['n_steps']:>6} {row['tau_max']:>4} {row['pc_alpha']:>7} "
            f"{row['precision']:>8.4f} {row['recall']:>8.4f} {row['edge_f1']:>8.4f} "
            f"{row['shd_style']:>4} {row['runtime_s']:>7.1f}s"
        )

    best = max(all_rows, key=rank_key)
    print(
        f"\nBEST CONFIG: n={best['n_steps']}, tau_max={best['tau_max']}, "
        f"pc_alpha={best['pc_alpha']} -> P={best['precision']:.4f}, "
        f"R={best['recall']:.4f}, F1={best['edge_f1']:.4f}, SHD={best['shd_style']}"
    )
    for n_steps in SIZES:
        best_n = max((r for r in all_rows if r["n_steps"] == n_steps), key=rank_key)
        print(
            f"BEST @ n={n_steps}: tau_max={best_n['tau_max']}, "
            f"pc_alpha={best_n['pc_alpha']} -> P={best_n['precision']:.4f}, "
            f"R={best_n['recall']:.4f}, F1={best_n['edge_f1']:.4f}, "
            f"SHD={best_n['shd_style']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

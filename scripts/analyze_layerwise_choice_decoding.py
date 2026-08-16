"""Decode risky choices from final-sentence hidden states, one layer at a time.

The analysis deliberately fixes reasoning position at the final sentence. It
therefore asks a single question: at which model layer is the participant's
choice most linearly accessible?

Two generalization tests are reported:
1. held-out participants
2. held-out decision problems

Example
-------
python analyze_layerwise_choice_decoding.py ^
  --state-dir path/to/qwen3.5-9b_bf16_sentence_end ^
  --output-dir path/to/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--c",
        type=float,
        default=0.01,
        help="Inverse L2 regularization strength, fixed across every layer.",
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def load_final_sentence_metadata(state_dir: Path) -> pd.DataFrame:
    sentences = pd.read_csv(state_dir / "sentences.csv")
    required = {
        "sentence_row",
        "trial_row",
        "participant_id",
        "problem_id",
        "choice",
        "sentence_index",
        "model_p_b",
    }
    missing = required.difference(sentences.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    final = (
        sentences.sort_values(["trial_row", "sentence_index"])
        .groupby("trial_row", sort=False)
        .tail(1)
        .sort_values("trial_row")
        .reset_index(drop=True)
    )
    if final["trial_row"].duplicated().any():
        raise ValueError("Expected exactly one final sentence per trial.")
    if not set(final["choice"].unique()).issubset({0, 1}):
        raise ValueError("Choice must be binary: 0 for A, 1 for B.")
    return final


def score_fold(y_true: np.ndarray, p_b: np.ndarray) -> dict[str, float]:
    predicted = (p_b >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "log_loss": log_loss(y_true, p_b, labels=[0, 1]),
        "brier": brier_score_loss(y_true, p_b),
        "roc_auc": roc_auc_score(y_true, p_b),
    }


def make_splits(
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    return list(splitter.split(np.zeros(len(y)), y, groups))


def fit_one_layer(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    layer: int,
    split_name: str,
    c_value: float,
    seed: int,
    max_iter: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for fold, (train_index, test_index) in enumerate(splits):
        # StandardScaler is inside the pipeline, so test-fold statistics never
        # leak into training.
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                penalty="l2",
                solver="liblinear",
                max_iter=max_iter,
                random_state=seed,
            ),
        )
        probe.fit(x[train_index], y[train_index])
        p_b = probe.predict_proba(x[test_index])[:, 1]
        metrics = score_fold(y[test_index], p_b)
        rows.append(
            {
                "split": split_name,
                "layer": layer,
                "fold": fold,
                "n_train": len(train_index),
                "n_test": len(test_index),
                **metrics,
            }
        )
    return rows


def aggregate_folds(fold_results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "balanced_accuracy", "log_loss", "brier", "roc_auc"]
    summary = (
        fold_results.groupby(["split", "layer"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def plot_layer_curves(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = {
        "participant": "Held-out participants",
        "problem": "Held-out decision problems",
    }
    colors = {"participant": "#2166AC", "problem": "#B2182B"}
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    for split_name in ["participant", "problem"]:
        subset = summary[summary["split"] == split_name].sort_values("layer")
        x = subset["layer"].to_numpy()
        mean = subset["balanced_accuracy_mean"].to_numpy()
        sd = subset["balanced_accuracy_std"].to_numpy()
        ax.plot(
            x,
            mean,
            color=colors[split_name],
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            label=labels[split_name],
        )
        ax.fill_between(x, mean - sd, mean + sd, color=colors[split_name], alpha=0.12)

    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.2, label="Chance")
    ax.set(
        xlabel="Qwen3.5-9B layer (0 = input embedding)",
        ylabel="Balanced accuracy",
        title="Where does choice become linearly decodable?",
        xlim=(0, 32),
        ylim=(0.45, 1.01),
    )
    ax.set_xticks(np.arange(0, 33, 4))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "layerwise_balanced_accuracy.svg")
    fig.savefig(output_dir / "layerwise_balanced_accuracy.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.state_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    final = load_final_sentence_metadata(args.state_dir)

    sentence_rows = final["sentence_row"].to_numpy(dtype=np.int64)
    y = final["choice"].to_numpy(dtype=np.int64)
    layer_count = int(manifest["layer_count"])

    split_specs = {
        "participant": final["participant_id"].to_numpy(),
        "problem": final["problem_id"].to_numpy(),
    }
    split_indices = {
        name: make_splits(y, groups, args.folds, args.seed)
        for name, groups in split_specs.items()
    }

    all_results: list[dict[str, float | int | str]] = []
    for layer in range(layer_count):
        layer_path = args.state_dir / f"layer_{layer:02d}.npy"
        matrix = np.load(layer_path, mmap_mode="r")
        x = np.asarray(matrix[sentence_rows], dtype=np.float32)
        for split_name, splits in split_indices.items():
            all_results.extend(
                fit_one_layer(
                    x,
                    y,
                    splits,
                    layer=layer,
                    split_name=split_name,
                    c_value=args.c,
                    seed=args.seed,
                    max_iter=args.max_iter,
                )
            )
        print(f"Completed layer {layer:02d}/{layer_count - 1:02d}", flush=True)

    fold_results = pd.DataFrame(all_results)
    summary = aggregate_folds(fold_results)
    fold_results.to_csv(args.output_dir / "layerwise_probe_folds.csv", index=False)
    summary.to_csv(args.output_dir / "layerwise_probe_summary.csv", index=False)

    native_metrics = score_fold(y, final["model_p_b"].to_numpy(dtype=float))
    best_rows = (
        summary.sort_values(
            ["split", "balanced_accuracy_mean"],
            ascending=[True, False],
        )
        .groupby("split", sort=False)
        .head(1)
    )
    report = {
        "model": manifest["model"],
        "representation": "final sentence-end hidden state",
        "trials": len(final),
        "participants": int(final["participant_id"].nunique()),
        "problems": int(final["problem_id"].nunique()),
        "layers": layer_count,
        "folds": args.folds,
        "seed": args.seed,
        "logistic_regression_c": args.c,
        "selection_metric": "mean cross-validated balanced accuracy",
        "native_model_metrics": native_metrics,
        "best_layers": {
            row["split"]: {
                "layer": int(row["layer"]),
                "balanced_accuracy_mean": float(row["balanced_accuracy_mean"]),
                "balanced_accuracy_std": float(row["balanced_accuracy_std"]),
                "log_loss_mean": float(row["log_loss_mean"]),
                "brier_mean": float(row["brier_mean"]),
            }
            for _, row in best_rows.iterrows()
        },
    }
    with (args.output_dir / "layerwise_probe_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)

    plot_layer_curves(summary, args.output_dir)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

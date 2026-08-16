"""Prepare Choice13K option descriptions and 12D Text2Decision targets.

The original Text2Decision training treats each option as one stimulus:
29,136 option descriptions from 14,568 binary-choice problems. This script
reproduces the original text format and decision-feature definitions without
depending on the legacy OpenAI embedding code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "maximum_gain",
    "minimum_gain",
    "maximum_loss",
    "minimum_loss",
    "sum_two_largest_gains",
    "probability_maximum_gain",
    "probability_minimum_gain",
    "probability_maximum_loss",
    "probability_minimum_loss",
    "probability_two_largest_gains",
    "expected_value",
    "entropy",
]
OUTCOME_SCALED_INDICES = [0, 1, 2, 3, 4, 10]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_duplicate_outcomes(option: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    probabilities: dict[float, float] = {}
    for probability, outcome in option:
        outcome = float(outcome)
        rounded_probability = round(float(probability), 4)
        probabilities[outcome] = probabilities.get(outcome, 0.0) + rounded_probability
    outcomes = np.asarray(list(probabilities), dtype=np.float64)
    probs = np.asarray([probabilities[value] for value in outcomes], dtype=np.float64)
    return probs, outcomes


def option_text(probabilities: np.ndarray, outcomes: np.ndarray) -> str:
    parts = [
        f"{float(value)} dollars with {float(probability * 100)} % chance"
        for probability, value in zip(probabilities, outcomes)
    ]
    return ", ".join(parts) + "."


def decision_features(probabilities: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    positive = outcomes >= 0
    negative = outcomes <= 0

    if np.all(outcomes <= 0):
        maximum_gain = minimum_gain = 0.0
        probability_maximum_gain = probability_minimum_gain = 0.0
    else:
        maximum_gain = float(outcomes.max())
        minimum_gain = float(outcomes[positive].min())
        probability_maximum_gain = float(probabilities[outcomes == maximum_gain].sum())
        probability_minimum_gain = float(probabilities[outcomes == minimum_gain].sum())

    if np.all(outcomes >= 0):
        maximum_loss = minimum_loss = 0.0
        probability_maximum_loss = probability_minimum_loss = 0.0
    else:
        maximum_loss = float(outcomes.min())
        minimum_loss = float(outcomes[negative].max())
        probability_maximum_loss = float(probabilities[outcomes == maximum_loss].sum())
        probability_minimum_loss = float(probabilities[outcomes == minimum_loss].sum())

    descending = np.sort(outcomes)[::-1]
    if len(descending) == 1 and descending[0] > 0:
        # Preserve the legacy definition for a deterministic positive option.
        sum_two_largest_gains = float(descending[0])
        probability_two_largest_gains = float(probabilities[0])
    elif len(descending) > 1 and np.sum(descending >= 0) > 1:
        two_largest = descending[:2]
        sum_two_largest_gains = float(two_largest.sum())
        probability_two_largest_gains = float(
            probabilities[np.isin(outcomes, two_largest)].sum()
        )
    else:
        sum_two_largest_gains = 0.0
        probability_two_largest_gains = 0.0

    expected_value = float(np.dot(outcomes, probabilities))
    nonzero = probabilities > 0
    entropy = float(-(probabilities[nonzero] * np.log2(probabilities[nonzero])).sum())

    return np.asarray(
        [
            maximum_gain,
            minimum_gain,
            maximum_loss,
            minimum_loss,
            sum_two_largest_gains,
            probability_maximum_gain,
            probability_minimum_gain,
            probability_maximum_loss,
            probability_minimum_loss,
            probability_two_largest_gains,
            expected_value,
            entropy,
        ],
        dtype=np.float32,
    )


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    problems = pd.read_json(args.data_path, orient="index")

    rows: list[dict[str, object]] = []
    targets: list[np.ndarray] = []
    # Preserve the source column order used by the original pandas melt: B, A.
    for problem_id, problem in problems.iterrows():
        for option_label in ["B", "A"]:
            probabilities, outcomes = merge_duplicate_outcomes(problem[option_label])
            features = decision_features(probabilities, outcomes)
            scaled = features.copy()
            scaled[OUTCOME_SCALED_INDICES] /= 1000.0
            rows.append(
                {
                    "stimulus_row": len(rows),
                    "problem_id": int(problem_id),
                    "option": option_label,
                    "stimulus_text": option_text(probabilities, outcomes),
                }
            )
            targets.append(scaled)

    metadata_path = args.output_dir / "stimuli.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    target_matrix = np.stack(targets).astype(np.float32)
    np.save(args.output_dir / "decision_targets.npy", target_matrix)

    manifest = {
        "format_version": 1,
        "source_file": args.data_path.name,
        "source_sha256": sha256_file(args.data_path),
        "problem_count": len(problems),
        "stimulus_count": len(rows),
        "stimulus_unit": "one risky option",
        "option_order": ["B", "A"],
        "feature_names": FEATURE_NAMES,
        "outcome_scaled_indices": OUTCOME_SCALED_INDICES,
        "outcome_scale_divisor": 1000.0,
        "target_shape": list(target_matrix.shape),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()





"""Combine reused x1 Choice13K states with newly extracted scale states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_STATES = (
    ROOT
    / "artifacts/text2decision/qwen35_layer15_legacy_exact"
    / "stimulus_layer_15.npy"
)
MULTISCALE_DIR = ROOT / "artifacts/text2decision/c13k_multiscale"
NEW_STATES = MULTISCALE_DIR / "new_scale_states/stimulus_layer_15.npy"
OUTPUT_STATES = MULTISCALE_DIR / "augmented_layer_15.npy"


def main() -> None:
    metadata = pd.read_csv(MULTISCALE_DIR / "augmented_stimuli.csv")
    original = np.load(ORIGINAL_STATES, mmap_mode="r")
    new = np.load(NEW_STATES, mmap_mode="r")
    x1_count = int((metadata["scale_factor"] == 1).sum())
    if x1_count != len(original):
        raise ValueError("x1 metadata count does not match original states")
    if len(metadata) - x1_count != len(new):
        raise ValueError("new-scale metadata count does not match new states")
    if not np.all(metadata.iloc[:x1_count]["scale_factor"] == 1):
        raise ValueError("Expected x1 block first in augmented metadata")
    if not np.all(metadata.iloc[x1_count:]["scale_factor"] > 1):
        raise ValueError("Expected new-scale blocks after x1 block")

    combined = np.lib.format.open_memmap(
        OUTPUT_STATES,
        mode="w+",
        dtype=np.float16,
        shape=(len(metadata), original.shape[1]),
    )
    combined[:x1_count] = original
    for start in range(0, len(new), 4096):
        end = min(start + 4096, len(new))
        combined[x1_count + start : x1_count + end] = new[start:end]
    combined.flush()
    report = {
        "shape": list(combined.shape),
        "dtype": str(combined.dtype),
        "x1_rows_reused": x1_count,
        "new_scale_rows": len(new),
        "row_order": "x1, x10, x100, x200",
    }
    (MULTISCALE_DIR / "combined_states_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Create scale-augmented Choice13K texts and 12D targets.

The original x1 states are reused. New Qwen states are needed for x10, x100,
and x200 because the monetary values in the stimulus text change.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_c13k_text2decision import OUTCOME_SCALED_INDICES


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "artifacts/text2decision/c13k_stimuli"
OUTPUT_DIR = ROOT / "artifacts/text2decision/c13k_multiscale"
FACTORS = [1.0, 10.0, 100.0, 200.0]
AMOUNT_PATTERN = re.compile(r"([-+]?\d+(?:\.\d+)?)\s+dollars")


def scaled_text(text: str, factor: float) -> str:
    def replace(match: re.Match[str]) -> str:
        value = float(match.group(1)) * factor
        return f"{value} dollars"

    return AMOUNT_PATTERN.sub(replace, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_DIR / "stimuli.csv")
    targets = np.load(SOURCE_DIR / "decision_targets.npy").astype(np.float32)

    frames: list[pd.DataFrame] = []
    target_blocks: list[np.ndarray] = []
    for factor in FACTORS:
        frame = source.copy()
        frame["scale_factor"] = factor
        frame["source_stimulus_row"] = frame["stimulus_row"]
        frame["stimulus_text"] = frame["stimulus_text"].map(
            lambda text: scaled_text(text, factor)
        )
        scaled_targets = targets.copy()
        scaled_targets[:, OUTCOME_SCALED_INDICES] *= factor
        frame["augmented_row"] = np.arange(
            len(frame) * len(frames), len(frame) * (len(frames) + 1)
        )
        frames.append(frame)
        target_blocks.append(scaled_targets)

    augmented = pd.concat(frames, ignore_index=True)
    augmented.to_csv(OUTPUT_DIR / "augmented_stimuli.csv", index=False)
    np.save(
        OUTPUT_DIR / "augmented_targets.npy",
        np.concatenate(target_blocks, axis=0),
    )
    new_scales = augmented[augmented["scale_factor"] > 1].copy()
    new_scales.to_csv(OUTPUT_DIR / "new_scale_stimuli.csv", index=False)

    manifest = {
        "source_stimuli": len(source),
        "scale_factors": FACTORS,
        "augmented_stimuli": len(augmented),
        "new_states_to_extract": len(new_scales),
        "split_group": (
            "original problem_id; both options and every scale stay together"
        ),
        "target_rule": (
            "monetary dimensions multiplied by scale factor; probability and "
            "entropy dimensions unchanged"
        ),
        "text_rule": "every numeric amount immediately before 'dollars' multiplied",
    }
    (OUTPUT_DIR / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

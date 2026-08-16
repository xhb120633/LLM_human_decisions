"""Run trajectory analysis with the signed-log multiscale checkpoint."""

from pathlib import Path

import analyze_text2decision_trajectories as analysis
from text2decision_transforms import signed_log_monetary


ROOT = Path(__file__).resolve().parents[1]
analysis.MODEL_PATH = (
    ROOT
    / "artifacts/text2decision/qwen35_layer15_text2decision_multiscale_log"
    / "TextDecisionModel_qwen_layer15.pt"
)
analysis.OUTPUT_DIR = (
    ROOT
    / "notebooks/results/representation"
    / "text2decision_multiscale_log_trajectories"
)
raw_option_target = analysis.option_target


def transformed_option_target(question: str, label: str):
    raw = raw_option_target(question, label)
    return signed_log_monetary(raw[None, :])[0]


analysis.option_target = transformed_option_target


if __name__ == "__main__":
    analysis.main()

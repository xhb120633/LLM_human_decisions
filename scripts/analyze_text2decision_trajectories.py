"""Interpret cumulative Think-Aloud states with the trained Text2Decision map.

The primary analysis deliberately excludes the explicit masked conclusion
sentence ("Option X") and holds out entire decision problems during choice
prediction.  This makes the tutorial example about information accumulated
during reasoning, rather than about reading a conclusion token.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_mutual_info_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_c13k_text2decision import (
    FEATURE_NAMES,
    OUTCOME_SCALED_INDICES,
    decision_features,
)
from train_qwen_text2decision import TextDecisionModel


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = (
    ROOT / "artifacts/qwen35_sentence_states/qwen3.5-9b_bf16_sentence_end"
)
ONSET_DIR = (
    ROOT / "artifacts/qwen35_sentence_states/qwen3.5-9b_bf16_question_onsets"
)
MODEL_PATH = (
    ROOT
    / "artifacts/text2decision/qwen35_layer15_text2decision"
    / "TextDecisionModel_qwen_layer15.pt"
)
OUTPUT_DIR = (
    ROOT / "notebooks/results/representation/text2decision_trajectories"
)
SEED = 2026

SHORT_FEATURE_NAMES = [
    "max gain",
    "min gain",
    "max loss",
    "min loss",
    "top-2 gains",
    "P(max gain)",
    "P(min gain)",
    "P(max loss)",
    "P(min loss)",
    "P(top-2 gains)",
    "expected value",
    "entropy",
]


def parse_option(question: str, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse one option from the standardized RiskyThoughts question text."""
    other = "B" if label == "A" else None
    if label == "A":
        match = re.search(r"Option A:\s*(.*?)\s*Option B:", question)
    else:
        match = re.search(r"Option B:\s*(.*?)(?:\s*$)", question)
    if not match:
        raise ValueError(f"Could not parse Option {label}: {question}")
    chunk = match.group(1)
    pairs = re.findall(
        r"([-+]?\d+(?:\.\d+)?)\s+dollars with\s+"
        r"([-+]?\d+(?:\.\d+)?)\s*%\s+chance",
        chunk,
    )
    if not pairs:
        raise ValueError(f"No outcomes parsed for Option {label}: {chunk}")
    outcomes = np.asarray([float(value) for value, _ in pairs], dtype=np.float64)
    probabilities = np.asarray(
        [float(probability) / 100.0 for _, probability in pairs],
        dtype=np.float64,
    )
    return probabilities, outcomes


def option_target(question: str, label: str) -> np.ndarray:
    probabilities, outcomes = parse_option(question, label)
    features = decision_features(probabilities, outcomes)
    features[OUTCOME_SCALED_INDICES] /= 1000.0
    return features


def infer_states(
    model: TextDecisionModel, states: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            batch = np.asarray(states[start : start + batch_size], dtype=np.float32)
            predictions.append(model(torch.from_numpy(batch)).numpy())
    return np.concatenate(predictions, axis=0)


def probability_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
    }


def grouped_predictions(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, list[dict[str, object]]]:
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )
    probability = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splitter.split(x, y, groups), start=1):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        )
        classifier.fit(x[train], y[train])
        fold_probability = classifier.predict_proba(x[test])[:, 1]
        probability[test] = fold_probability
        held_questions = sorted(set(groups[test]))
        for row, p in zip(test, fold_probability):
            fold_rows.append(
                {
                    "row": int(row),
                    "fold": fold,
                    "held_out_questions": "|".join(held_questions),
                    "probability_B": float(p),
                }
            )
    if np.isnan(probability).any():
        raise RuntimeError("Some trials did not receive an out-of-fold prediction")
    return probability, fold_rows


def interpolate_trajectory(values: np.ndarray, fraction: float) -> np.ndarray:
    if len(values) == 1 or fraction <= 0:
        return values[0]
    location = fraction * (len(values) - 1)
    lower = int(np.floor(location))
    upper = int(np.ceil(location))
    if lower == upper:
        return values[lower]
    weight = location - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    sentences = pd.read_csv(STATE_DIR / "sentences.csv")
    trial_map = pd.read_csv(ONSET_DIR / "trial_question_map.csv").sort_values(
        "trial_row"
    )
    sentence_states = np.load(STATE_DIR / "layer_15.npy", mmap_mode="r")
    onset_states = np.load(
        ONSET_DIR / "reasoning_onset_layer_15.npy", mmap_mode="r"
    )
    onset_metadata = pd.read_csv(ONSET_DIR / "onsets.csv")
    if len(sentences) != len(sentence_states):
        raise ValueError("Sentence metadata and states differ in length")
    if len(onset_metadata) != len(onset_states):
        raise ValueError("Onset metadata and states differ in length")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = TextDecisionModel(
        text_dim=int(checkpoint["text_dim"]),
        decision_dim=int(checkpoint["decision_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    target_mean = checkpoint["target_mean"].numpy()
    target_std = checkpoint["target_std"].numpy()

    sentence_z = infer_states(model, sentence_states)
    onset_z = infer_states(model, onset_states)
    onset_by_question = {
        question_id: onset_z[int(onset_row)]
        for question_id, onset_row in zip(
            onset_metadata["question_id"], onset_metadata["onset_row"]
        )
    }

    question_rows = trial_map.drop_duplicates("question_id")
    anchors: dict[str, dict[str, np.ndarray]] = {}
    anchor_records: list[dict[str, object]] = []
    for row in question_rows.itertuples():
        a_raw = option_target(row.question, "A")
        b_raw = option_target(row.question, "B")
        a = (a_raw - target_mean) / target_std
        b = (b_raw - target_mean) / target_std
        midpoint = (a + b) / 2.0
        difference = b - a
        distance = float(np.linalg.norm(difference))
        unit = difference / distance
        anchors[row.question_id] = {
            "A": a,
            "B": b,
            "midpoint": midpoint,
            "unit": unit,
        }
        for label, vector in [("A", a), ("B", b)]:
            anchor_records.append(
                {
                    "question_id": row.question_id,
                    "option": label,
                    **{
                        f"z_{name}": float(value)
                        for name, value in zip(FEATURE_NAMES, vector)
                    },
                }
            )
    pd.DataFrame(anchor_records).to_csv(
        OUTPUT_DIR / "question_option_anchors_z.csv", index=False
    )

    sentences = sentences.merge(
        trial_map[
            [
                "trial_row",
                "question_id",
                "question",
                "choice",
                "persona",
            ]
        ],
        on="trial_row",
        suffixes=("", "_map"),
        validate="many_to_one",
    )
    for index, name in enumerate(FEATURE_NAMES):
        sentences[f"z_{name}"] = sentence_z[:, index]
    sentence_evidence = np.empty(len(sentences), dtype=np.float32)
    for question_id, indices in sentences.groupby("question_id").groups.items():
        midpoint = anchors[question_id]["midpoint"]
        unit = anchors[question_id]["unit"]
        sentence_evidence[np.asarray(indices)] = (
            sentence_z[np.asarray(indices)] - midpoint
        ) @ unit
    sentences["choice_evidence_B"] = sentence_evidence
    sentences["is_masked_conclusion"] = sentences["sentence_text"].str.contains(
        "Option X", case=False, regex=False
    )

    trial_records: list[dict[str, object]] = []
    trajectory_z: dict[int, np.ndarray] = {}
    trajectory_evidence: dict[int, np.ndarray] = {}
    preconclusion_rows: dict[int, pd.DataFrame] = {}
    for trial in trial_map.itertuples():
        rows = sentences[sentences["trial_row"] == trial.trial_row].sort_values(
            "sentence_index"
        )
        conclusion_positions = np.flatnonzero(
            rows["is_masked_conclusion"].to_numpy()
        )
        cutoff = int(conclusion_positions[0]) if len(conclusion_positions) else len(rows)
        primary = rows.iloc[:cutoff].copy()
        if primary.empty:
            primary = rows.iloc[:1].copy()
        z = np.vstack(
            [
                onset_by_question[trial.question_id],
                primary[[f"z_{name}" for name in FEATURE_NAMES]].to_numpy(),
            ]
        )
        anchor = anchors[trial.question_id]
        evidence = (z - anchor["midpoint"]) @ anchor["unit"]
        delta_z = np.diff(z, axis=0)
        delta_evidence = np.diff(evidence)
        trajectory_z[trial.trial_row] = z
        trajectory_evidence[trial.trial_row] = evidence
        preconclusion_rows[trial.trial_row] = primary

        progress = np.linspace(0.0, 1.0, len(evidence))
        auc = float(np.trapezoid(evidence, progress)) if len(evidence) > 1 else 0.0
        slope = (
            float(np.polyfit(progress, evidence, 1)[0])
            if len(evidence) > 1
            else 0.0
        )
        orthogonal = delta_z - delta_evidence[:, None] * anchor["unit"][None, :]
        record: dict[str, object] = {
            "trial_row": int(trial.trial_row),
            "participant_id": int(trial.participant_id),
            "question_id": trial.question_id,
            "choice": int(trial.choice),
            "persona": trial.persona,
            "preconclusion_sentences": int(len(primary)),
            "q_onset": float(evidence[0]),
            "q_endpoint": float(evidence[-1]),
            "q_change": float(evidence[-1] - evidence[0]),
            "q_auc": auc,
            "q_slope": slope,
            "max_abs_delta_q": float(np.max(np.abs(delta_evidence))),
            "path_length": float(np.linalg.norm(delta_z, axis=1).sum()),
            "orthogonal_path_length": float(
                np.linalg.norm(orthogonal, axis=1).sum()
            ),
        }
        for index, name in enumerate(FEATURE_NAMES):
            record[f"onset_z_{name}"] = float(z[0, index])
            record[f"endpoint_z_{name}"] = float(z[-1, index])
            record[f"change_z_{name}"] = float(z[-1, index] - z[0, index])
            record[f"total_abs_change_z_{name}"] = float(
                np.abs(delta_z[:, index]).sum()
            )
        trial_records.append(record)
    trials = pd.DataFrame(trial_records)
    trials.to_csv(OUTPUT_DIR / "trial_trajectory_summary.csv", index=False)
    sentences.to_csv(OUTPUT_DIR / "sentence_decision_states.csv", index=False)

    y = trials["choice"].to_numpy()
    groups = trials["question_id"].to_numpy()
    feature_sets: dict[str, np.ndarray] = {
        "option-axis endpoint": trials[["q_endpoint"]].to_numpy(),
        "12D endpoint": trials[
            [f"endpoint_z_{name}" for name in FEATURE_NAMES]
        ].to_numpy(),
        "trajectory summary": trials[
            [
                "q_endpoint",
                "q_change",
                "q_auc",
                "q_slope",
                "max_abs_delta_q",
                "path_length",
                "orthogonal_path_length",
            ]
        ].to_numpy(),
    }
    prediction_rows: list[dict[str, object]] = []
    fold_rows_all: list[dict[str, object]] = []
    for label, x in feature_sets.items():
        probability, fold_rows = grouped_predictions(x, y, groups)
        metric = probability_metrics(y, probability)
        prediction_rows.append({"model": label, **metric})
        for row in fold_rows:
            row["model"] = label
            row["trial_row"] = int(trials.iloc[int(row["row"])]["trial_row"])
            fold_rows_all.append(row)

    progress_rows: list[dict[str, object]] = []
    for fraction in [0.0, 0.25, 0.50, 0.75, 1.0]:
        x_progress = np.vstack(
            [
                interpolate_trajectory(trajectory_z[int(row.trial_row)], fraction)
                for row in trials.itertuples()
            ]
        )
        probability, _ = grouped_predictions(x_progress, y, groups)
        progress_rows.append(
            {
                "progress": fraction,
                "representation": "12D state",
                **probability_metrics(y, probability),
            }
        )
        q_progress = np.asarray(
            [
                interpolate_trajectory(
                    trajectory_evidence[int(row.trial_row)][:, None], fraction
                )[0]
                for row in trials.itertuples()
            ]
        )[:, None]
        probability, _ = grouped_predictions(q_progress, y, groups)
        progress_rows.append(
            {
                "progress": fraction,
                "representation": "option-axis scalar",
                **probability_metrics(y, probability),
            }
        )
    prediction_summary = pd.DataFrame(prediction_rows)
    progress_summary = pd.DataFrame(progress_rows)
    prediction_summary.to_csv(
        OUTPUT_DIR / "choice_prediction_summary.csv", index=False
    )
    progress_summary.to_csv(
        OUTPUT_DIR / "choice_prediction_by_progress.csv", index=False
    )
    pd.DataFrame(fold_rows_all).to_csv(
        OUTPUT_DIR / "choice_prediction_folds.csv", index=False
    )

    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    styles = {
        "12D state": ("#2C6E9B", "o"),
        "option-axis scalar": ("#B45F3C", "s"),
    }
    for label, frame in progress_summary.groupby("representation", sort=False):
        color, marker = styles[label]
        axis.plot(
            frame["progress"] * 100,
            frame["balanced_accuracy"],
            marker=marker,
            linewidth=2.4,
            markersize=7,
            color=color,
            label=label,
        )
    axis.axhline(0.5, color="#777777", linewidth=1.2, linestyle="--")
    axis.set(
        xlabel="Reasoning available before the masked conclusion (%)",
        ylabel="Balanced accuracy\n(held-out questions)",
        ylim=(0.40, max(0.72, progress_summary["balanced_accuracy"].max() + 0.04)),
        title="Does the evolving representation predict the eventual choice?",
    )
    axis.legend(frameon=False)
    sns.despine(ax=axis)
    fig.tight_layout()
    save_figure(fig, "heldout_choice_prediction_over_reasoning")

    q04 = trials[trials["question_id"] == "Q04"].copy()
    persona_progress: list[dict[str, object]] = []
    for row in q04.itertuples():
        for fraction in np.linspace(0.0, 1.0, 9):
            evidence = interpolate_trajectory(
                trajectory_evidence[int(row.trial_row)][:, None], fraction
            )[0]
            persona_progress.append(
                {
                    "persona": row.persona,
                    "choice": row.choice,
                    "progress": fraction,
                    "evidence": float(evidence),
                }
            )
    persona_frame = pd.DataFrame(persona_progress)
    persona_mean = (
        persona_frame.groupby(["persona", "progress"], as_index=False)
        .agg(mean=("evidence", "mean"), sem=("evidence", "sem"))
    )
    fig, axis = plt.subplots(figsize=(9.6, 5.6))
    palette = sns.color_palette("colorblind", n_colors=persona_mean["persona"].nunique())
    for color, (persona, frame) in zip(
        palette, persona_mean.groupby("persona", sort=True)
    ):
        x = frame["progress"].to_numpy() * 100
        mean = frame["mean"].to_numpy()
        sem = frame["sem"].fillna(0).to_numpy()
        axis.plot(x, mean, linewidth=2.2, label=persona, color=color)
        axis.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.12)
    axis.axhline(0, color="#777777", linewidth=1.2, linestyle="--")
    axis.set(
        xlabel="Reasoning progress before the masked conclusion (%)",
        ylabel="Position on the A-to-B option axis\n(negative = A, positive = B)",
        title="Q04: persona-conditioned reasoning follows different semantic paths",
    )
    axis.legend(frameon=False, fontsize=10, loc="best")
    sns.despine(ax=axis)
    fig.tight_layout()
    save_figure(fig, "q04_persona_option_axis")

    candidates = []
    for persona, choice in [("Rational Decision-Maker", 1), ("Risk-Averse Decision-Maker", 0)]:
        frame = q04[(q04["persona"] == persona) & (q04["choice"] == choice)]
        if frame.empty:
            continue
        median_length = frame["preconclusion_sentences"].median()
        selected = frame.iloc[
            np.argmin(np.abs(frame["preconclusion_sentences"] - median_length))
        ]
        candidates.append(selected)

    if len(candidates) == 2:
        all_updates = []
        trial_updates = []
        for selected in candidates:
            trial_row = int(selected["trial_row"])
            z = trajectory_z[trial_row]
            unit = anchors["Q04"]["unit"]
            updates = np.diff(z, axis=0) * unit[None, :]
            all_updates.append(updates)
            trial_updates.append((selected, updates))
        importance = np.sum(
            np.abs(np.vstack(all_updates)), axis=0
        )
        top_dimensions = np.argsort(importance)[-8:]
        vmax = max(
            float(np.abs(updates[:, top_dimensions]).max())
            for _, updates in trial_updates
        )
        fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.6), sharex=True)
        for axis, (selected, updates) in zip(axes, trial_updates):
            trial_row = int(selected["trial_row"])
            text_rows = preconclusion_rows[trial_row]
            short_sentences = [
                text if len(text) <= 72 else text[:69] + "..."
                for text in text_rows["sentence_text"].tolist()
            ]
            sns.heatmap(
                updates[:, top_dimensions].T,
                ax=axis,
                cmap="vlag",
                center=0,
                vmin=-vmax,
                vmax=vmax,
                yticklabels=[SHORT_FEATURE_NAMES[i] for i in top_dimensions],
                xticklabels=[str(i + 1) for i in range(len(short_sentences))],
                cbar=axis is axes[-1],
                cbar_kws={"label": "Sentence update toward B (+) or A (-)"},
            )
            choice_label = "B" if int(selected["choice"]) == 1 else "A"
            axis.set_title(f"{selected['persona']} - eventual choice {choice_label}")
            axis.set_ylabel("")
            pd.DataFrame(
                {
                    "sentence_number": np.arange(1, len(short_sentences) + 1),
                    "sentence_text": text_rows["sentence_text"].tolist(),
                }
            ).to_csv(
                OUTPUT_DIR
                / f"q04_trial_{trial_row}_sentences.csv",
                index=False,
            )
        axes[-1].set_xlabel("Sentence number (full text saved alongside figure)")
        fig.suptitle(
            "Which inferred decision concepts move after each sentence?",
            y=1.01,
        )
        fig.tight_layout()
        save_figure(fig, "q04_sentence_concept_updates")

    cluster_x = np.hstack(
        [
            trials[[f"change_z_{name}" for name in FEATURE_NAMES]].to_numpy(),
            trials[
                [f"total_abs_change_z_{name}" for name in FEATURE_NAMES]
            ].to_numpy(),
        ]
    )
    cluster_x = StandardScaler().fit_transform(cluster_x)
    cluster_diagnostics: list[dict[str, object]] = []
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(
            cluster_x
        )
        cluster_diagnostics.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(cluster_x, labels)),
                "AMI_question": float(
                    adjusted_mutual_info_score(trials["question_id"], labels)
                ),
                "AMI_persona": float(
                    adjusted_mutual_info_score(trials["persona"], labels)
                ),
                "AMI_choice": float(
                    adjusted_mutual_info_score(trials["choice"], labels)
                ),
            }
        )
    pd.DataFrame(cluster_diagnostics).to_csv(
        OUTPUT_DIR / "trajectory_cluster_diagnostics.csv", index=False
    )

    report = {
        "n_trials": int(len(trials)),
        "n_sentences": int(len(sentences)),
        "n_questions": int(trials["question_id"].nunique()),
        "explicit_conclusion_excluded": True,
        "endpoint_definition": "last cumulative sentence state before first Option X sentence",
        "prediction_split": "5-fold StratifiedGroupKFold; exact question held out",
        "domain_shift_diagnostics": {
            "sentence_z_abs_gt_3_fraction": float(np.mean(np.abs(sentence_z) > 3)),
            "sentence_z_abs_gt_5_fraction": float(np.mean(np.abs(sentence_z) > 5)),
            "onset_z_abs_gt_3_fraction": float(np.mean(np.abs(onset_z) > 3)),
            "sentence_z_dimension_mean": sentence_z.mean(axis=0).tolist(),
            "sentence_z_dimension_std": sentence_z.std(axis=0).tolist(),
        },
        "prediction_summary": prediction_rows,
        "prediction_by_progress": progress_rows,
        "cluster_diagnostics": cluster_diagnostics,
    }
    (OUTPUT_DIR / "analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()



from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
SENTENCE_PATH = (
    PROJECT
    / "notebooks/results/representation/text2decision_multiscale_log_trajectories/"
    / "sentence_decision_states.csv"
)
OUTPUT_DIR = PROJECT / "notebooks/results/minimal_discovery_loop"
MODEL = "deepseek-v4-pro"
PARTICIPANT_ID = 69576
TRAIN_QUESTIONS = [f"Q{i:02d}" for i in range(1, 12)]
VALIDATION_QUESTIONS = [f"Q{i:02d}" for i in range(12, 16)]
TEST_QUESTIONS = [f"Q{i:02d}" for i in range(16, 20)]

FEATURE_LIBRARY = {
    "ev_diff": "expected value of B minus expected value of A",
    "sd_diff": "outcome standard deviation of B minus A",
    "worst_diff": "worst outcome of B minus A",
    "best_diff": "best outcome of B minus A",
    "p_best_diff": "probability of the best outcome in B minus A",
    "p_zero_diff": "probability of zero in B minus A",
    "certain_diff": "certainty indicator for B minus A",
}

LOTTERY_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?) dollars with (\d+(?:\.\d+)?) % chance")
OPTION = r"option\s+[ab]"
DECISIVE = (
    r"(?:choose|chose|select|pick|lean(?:ing)?|inclined|opt(?:ing)?|"
    r"go with|would take|will take|my choice|my decision|prefer)"
)
EXPLICIT_PATTERN = re.compile(
    rf"(?i)(?:{OPTION}.{{0,120}}\b{DECISIVE}\b|"
    rf"\b{DECISIVE}\b.{{0,120}}{OPTION}|"
    rf"\b{OPTION}\b.{{0,80}}(?:appealing|attractive).{{0,40}}(?:to me|for me))"
)


def is_explicit_claim(text: str) -> bool:
    return "Option X" in str(text) or bool(EXPLICIT_PATTERN.search(str(text)))


def preference_free_prefix(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("sentence_index")
    kept = []
    for sentence in ordered["sentence_text"].fillna("").astype(str):
        if is_explicit_claim(sentence):
            break
        kept.append(sentence)
    return " ".join(kept)


def parse_option(question: str, label: str) -> dict[str, float]:
    segment = question.split(f"Option {label}:", 1)[1]
    if label == "A":
        segment = segment.split("Option B:", 1)[0]
    pairs = [
        (float(value), float(probability) / 100)
        for value, probability in LOTTERY_PATTERN.findall(segment)
    ]
    outcomes = np.asarray([value for value, _ in pairs], dtype=float)
    probabilities = np.asarray([probability for _, probability in pairs], dtype=float)
    probabilities = probabilities / probabilities.sum()
    ev = float(np.sum(probabilities * outcomes))
    sd = float(np.sqrt(np.sum(probabilities * (outcomes - ev) ** 2)))
    best_index = int(np.argmax(outcomes))
    return {
        "ev": ev,
        "sd": sd,
        "worst": float(outcomes.min()),
        "best": float(outcomes.max()),
        "p_best": float(probabilities[best_index]),
        "p_zero": float(probabilities[np.isclose(outcomes, 0)].sum()),
        "certain": float(len(outcomes) == 1 or np.isclose(probabilities.max(), 1)),
    }


def load_participant_trials() -> pd.DataFrame:
    sentences = pd.read_csv(SENTENCE_PATH)
    person = sentences.loc[sentences["participant_id"] == PARTICIPANT_ID].copy()
    records = []
    for trial_row, frame in person.groupby("trial_row", sort=False):
        first = frame.iloc[0]
        option_a = parse_option(first["question"], "A")
        option_b = parse_option(first["question"], "B")
        records.append(
            {
                "trial_row": int(trial_row),
                "question_id": first["question_id"],
                "question": first["question"],
                "choice": int(first["choice"]),
                "choice_label": "B" if int(first["choice"]) else "A",
                "think_aloud": preference_free_prefix(frame),
                **{
                    f"{name}_diff": option_b[name] - option_a[name]
                    for name in option_a
                },
            }
        )
    trials = pd.DataFrame(records).sort_values("question_id").reset_index(drop=True)
    assert trials["question_id"].tolist() == TRAIN_QUESTIONS + VALIDATION_QUESTIONS + TEST_QUESTIONS
    return trials


def split_trials(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": trials[trials["question_id"].isin(TRAIN_QUESTIONS)].copy(),
        "validation": trials[trials["question_id"].isin(VALIDATION_QUESTIONS)].copy(),
        "test": trials[trials["question_id"].isin(TEST_QUESTIONS)].copy(),
    }


def prompt_examples(frame: pd.DataFrame) -> list[dict]:
    examples = []
    for row in frame.itertuples():
        examples.append(
            {
                "question_id": row.question_id,
                "decision_problem": row.question,
                "observed_choice": row.choice_label,
                "think_aloud_before_explicit_choice": row.think_aloud[:1200],
            }
        )
    return examples


def proposal_messages(train: pd.DataFrame) -> list[dict]:
    schema = {
        "candidates": [
            {
                "name": "short descriptive name",
                "features": ["one or more allowed feature names"],
                "process_hypothesis": "one falsifiable sentence",
                "trace_evidence": "brief evidence from the training think-aloud",
            }
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "You help search a restricted cognitive-model space. Propose candidate "
                "computational accounts, not narratives. Python will fit coefficients and "
                "evaluate held-out choices. Use only allowed features and return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Propose exactly four distinct candidate models for this participant. "
                "The final explicit choice sentence has been removed from each think-aloud.\n\n"
                f"Allowed feature library:\n{json.dumps(FEATURE_LIBRARY, indent=2)}\n\n"
                f"Training records:\n{json.dumps(prompt_examples(train), indent=2)}\n\n"
                f"Required schema:\n{json.dumps(schema, indent=2)}"
            ),
        },
    ]


def revision_messages(
    train: pd.DataFrame,
    candidates: list[dict],
    validation_results: list[dict],
) -> list[dict]:
    compact_results = [
        {
            "name": row["name"],
            "features": row["features"],
            "validation_log_loss": row["validation_log_loss"],
            "validation_balanced_accuracy": row["validation_balanced_accuracy"],
            "validation_errors": row["validation_errors"],
        }
        for row in validation_results
    ]
    return [
        {
            "role": "system",
            "content": (
                "Revise a restricted cognitive-model search using validation feedback. "
                "Do not inspect or speculate about the test set. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Propose exactly three revised candidates. You may retain one candidate, "
                "remove an ingredient, or combine ingredients when the process evidence supports it.\n\n"
                f"Allowed feature library:\n{json.dumps(FEATURE_LIBRARY, indent=2)}\n\n"
                f"Training records:\n{json.dumps(prompt_examples(train), indent=2)}\n\n"
                f"Previous candidates:\n{json.dumps(candidates, indent=2)}\n\n"
                f"Validation feedback:\n{json.dumps(compact_results, indent=2)}\n\n"
                "Return {\"candidates\": [...]} with fields name, features, "
                "process_hypothesis, trace_evidence, and revision_reason."
            ),
        },
    ]


def call_deepseek(messages: list[dict]) -> dict:
    load_dotenv(PROJECT / ".env")
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing from .env")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1800,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    return {
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
        "content": json.loads(response.choices[0].message.content),
    }


def validate_candidates(payload: dict, expected_count: int) -> list[dict]:
    candidates = []
    seen = set()
    allowed = set(FEATURE_LIBRARY)
    for raw in payload.get("candidates", []):
        features = tuple(dict.fromkeys(raw.get("features", [])))
        if not features or not set(features).issubset(allowed) or features in seen:
            continue
        seen.add(features)
        candidates.append(
            {
                "name": str(raw.get("name", "candidate"))[:80],
                "features": list(features),
                "process_hypothesis": str(raw.get("process_hypothesis", ""))[:500],
                "trace_evidence": str(raw.get("trace_evidence", ""))[:500],
                "revision_reason": str(raw.get("revision_reason", ""))[:500],
            }
        )
    if len(candidates) < expected_count:
        raise ValueError(f"Expected {expected_count} valid candidates, received {len(candidates)}")
    return candidates[:expected_count]


def fit_and_score(spec: dict, train: pd.DataFrame, evaluation: pd.DataFrame) -> dict:
    features = spec["features"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=3000),
    )
    model.fit(train[features], train["choice"])
    probability = model.predict_proba(evaluation[features])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    errors = []
    for row, p_b, predicted in zip(evaluation.itertuples(), probability, prediction):
        if predicted != row.choice:
            errors.append(
                {
                    "question_id": row.question_id,
                    "actual": row.choice_label,
                    "predicted": "B" if predicted else "A",
                    "p_B": round(float(p_b), 3),
                }
            )
    return {
        **spec,
        "n_parameters": len(features) + 1,
        "probability_B": [float(value) for value in probability],
        "balanced_accuracy": float(balanced_accuracy_score(evaluation["choice"], prediction)),
        "log_loss": float(log_loss(evaluation["choice"], probability, labels=[0, 1])),
        "errors": errors,
    }


def validation_record(result: dict) -> dict:
    return {
        "name": result["name"],
        "features": result["features"],
        "process_hypothesis": result["process_hypothesis"],
        "trace_evidence": result["trace_evidence"],
        "validation_balanced_accuracy": result["balanced_accuracy"],
        "validation_log_loss": result["log_loss"],
        "validation_errors": result["errors"],
    }


def run_loop() -> dict:
    trials = load_participant_trials()
    splits = split_trials(trials)

    proposal_response = call_deepseek(proposal_messages(splits["train"]))
    initial = validate_candidates(proposal_response["content"], expected_count=4)
    initial_validation = [
        validation_record(fit_and_score(spec, splits["train"], splits["validation"]))
        for spec in initial
    ]

    revision_response = call_deepseek(
        revision_messages(splits["train"], initial, initial_validation)
    )
    revised = validate_candidates(revision_response["content"], expected_count=3)
    revised_validation = [
        validation_record(fit_and_score(spec, splits["train"], splits["validation"]))
        for spec in revised
    ]

    all_validation = initial_validation + revised_validation
    winner = min(all_validation, key=lambda row: row["validation_log_loss"])
    final_spec = {
        key: winner[key]
        for key in ["name", "features", "process_hypothesis", "trace_evidence"]
    }
    train_plus_validation = pd.concat(
        [splits["train"], splits["validation"]], ignore_index=True
    )
    final_test = fit_and_score(final_spec, train_plus_validation, splits["test"])

    baselines = []
    for spec in [
        {
            "name": "expected-value baseline",
            "features": ["ev_diff"],
            "process_hypothesis": "choices track expected value",
            "trace_evidence": "pre-registered baseline",
        },
        {
            "name": "all-feature baseline",
            "features": list(FEATURE_LIBRARY),
            "process_hypothesis": "unrestricted linear combination of the library",
            "trace_evidence": "capacity baseline",
        },
    ]:
        baselines.append(fit_and_score(spec, train_plus_validation, splits["test"]))

    return {
        "design": {
            "participant_id": PARTICIPANT_ID,
            "train_questions": TRAIN_QUESTIONS,
            "validation_questions": VALIDATION_QUESTIONS,
            "test_questions": TEST_QUESTIONS,
            "test_access": "opened once after candidate selection",
        },
        "proposal": proposal_response,
        "initial_validation": initial_validation,
        "revision": revision_response,
        "revised_validation": revised_validation,
        "selected_model": final_spec,
        "test_result": final_test,
        "test_baselines": baselines,
    }


def save_outputs(result: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "minimal_discovery_run.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    validation_rows = result["initial_validation"] + result["revised_validation"]
    pd.DataFrame(validation_rows).drop(
        columns=["validation_errors"], errors="ignore"
    ).to_csv(OUTPUT_DIR / "validation_results.csv", index=False)
    test_rows = [result["test_result"], *result["test_baselines"]]
    pd.DataFrame(
        [
            {
                "model": row["name"],
                "features": ", ".join(row["features"]),
                "n_parameters": row["n_parameters"],
                "balanced_accuracy": row["balanced_accuracy"],
                "log_loss": row["log_loss"],
            }
            for row in test_rows
        ]
    ).to_csv(OUTPUT_DIR / "test_results.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()
    cache = OUTPUT_DIR / "minimal_discovery_run.json"
    if args.use_cache and cache.exists():
        result = json.loads(cache.read_text(encoding="utf-8"))
    else:
        result = run_loop()
        save_outputs(result)
    print(json.dumps({
        "selected_model": result["selected_model"],
        "test_result": {
            "balanced_accuracy": result["test_result"]["balanced_accuracy"],
            "log_loss": result["test_result"]["log_loss"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "notebooks/data/behavioral_expanded_public_slice.csv"
OUTPUT_DIR = PROJECT / "notebooks/results/clean_icl_balance_demo"
MODEL = "deepseek-v4-pro"
PARTICIPANTS = ["P025", "P026"]
TARGET_TRIALS = list(range(21, 41))
HISTORY_K = 8
SEED = 20260812


def logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def extract_probability_b(response) -> float:
    items = response.choices[0].logprobs.content[0].top_logprobs
    groups = {"A": [], "B": []}
    for item in items:
        label = item.token.strip()
        if label in groups:
            groups[label].append(float(item.logprob))
    if not groups["A"] or not groups["B"]:
        raise ValueError("A and B must both appear in top_logprobs")
    label_lp = {label: logsumexp(values) for label, values in groups.items()}
    return math.exp(label_lp["B"] - logsumexp(list(label_lp.values())))


def swap_label(label: str) -> str:
    return "B" if label == "A" else "A"


def make_messages(target: pd.Series, history: pd.DataFrame, swap: bool) -> list[dict]:
    blocks = []
    for _, row in history.sort_values("trial_index").iterrows():
        lottery_a, lottery_b = row["lottery_A"], row["lottery_B"]
        choice = row["choice"]
        if swap:
            lottery_a, lottery_b = lottery_b, lottery_a
            choice = swap_label(choice)
        blocks.append(
            f"Earlier trial {int(row['trial_index'])}:\n"
            f"Option A: {lottery_a}\n"
            f"Option B: {lottery_b}\n"
            f"Observed choice: {choice}"
        )

    target_a, target_b = target["lottery_A"], target["lottery_B"]
    if swap:
        target_a, target_b = target_b, target_a

    history_text = ""
    if blocks:
        history_text = (
            "Here are earlier choices from the same participant:\n\n"
            + "\n\n".join(blocks)
            + "\n\n"
        )
    return [
        {
            "role": "system",
            "content": (
                "Answer with exactly one token: A or B. Option letters are arbitrary. "
                "Use the option content and any earlier choices to predict the participant."
            ),
        },
        {
            "role": "user",
            "content": (
                history_text
                + "Predict this participant's choice on a new trial.\n"
                + f"Option A: {target_a}\n"
                + f"Option B: {target_b}\n"
                + "Answer:"
            ),
        },
    ]


def call(client: OpenAI, messages: list[dict], attempts: int = 4):
    for attempt in range(attempts):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=1,
                max_tokens=1,
                logprobs=True,
                top_logprobs=20,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)


def logit(probability: float) -> float:
    probability = float(np.clip(probability, 1e-8, 1 - 1e-8))
    return math.log(probability / (1 - probability))


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def balanced_history(history: pd.DataFrame, participant: str) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + int(participant[1:]))
    selected = []
    for label in ["A", "B"]:
        pool = history[history["choice"] == label]
        chosen = rng.choice(pool.index.to_numpy(), size=HISTORY_K // 2, replace=False)
        selected.extend(chosen.tolist())
    return history.loc[selected].sort_values("trial_index")


def metrics(frame: pd.DataFrame, probability: str) -> dict:
    y = (frame["actual_choice"] == "B").astype(int).to_numpy()
    p = frame[probability].to_numpy(dtype=float)
    return {
        "n": len(frame),
        "A_targets": int((y == 0).sum()),
        "B_targets": int((y == 1).sum()),
        "accuracy": float(((p >= 0.5) == y).mean()),
        "log_loss": float(np.mean(-(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))),
        "brier": float(np.mean((p - y) ** 2)),
        "mean_p_B": float(p.mean()),
    }


def main() -> None:
    load_dotenv(PROJECT / ".env")
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    data = pd.read_csv(DATA_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT_DIR / "api_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[(row["participant_id"], row["target_trial"], row["condition"], row["swap"])] = row

    rows = []
    for participant in PARTICIPANTS:
        person = data[data["participant_id"] == participant].sort_values("trial_index")
        eligible = person[person["trial_index"] <= 20]
        histories = {
            "zero_shot": eligible.iloc[0:0],
            "unbalanced_history": eligible.head(HISTORY_K),
            "balanced_history": balanced_history(eligible, participant),
        }
        for condition, history in histories.items():
            for trial in TARGET_TRIALS:
                target = person[person["trial_index"] == trial].iloc[0]
                probabilities = {}
                for swap in [False, True]:
                    key = (participant, trial, condition, swap)
                    if key in cache:
                        record = cache[key]
                    else:
                        response = call(client, make_messages(target, history, swap))
                        p_surface_b = extract_probability_b(response)
                        record = {
                            "participant_id": participant,
                            "target_trial": trial,
                            "condition": condition,
                            "swap": swap,
                            "p_surface_B": p_surface_b,
                            "model": response.model,
                        }
                        with cache_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    probabilities[swap] = float(record["p_surface_B"])

                p_b_original = probabilities[False]
                p_b_from_swapped = 1 - probabilities[True]
                p_b_counterbalanced = sigmoid(
                    (logit(p_b_original) + logit(p_b_from_swapped)) / 2
                )
                rows.append(
                    {
                        "participant_id": participant,
                        "target_trial": trial,
                        "actual_choice": target["choice"],
                        "condition": condition,
                        "history_A": int((history["choice"] == "A").sum()),
                        "history_B": int((history["choice"] == "B").sum()),
                        "p_B_original": p_b_original,
                        "p_B_from_swapped": p_b_from_swapped,
                        "p_B_counterbalanced": p_b_counterbalanced,
                    }
                )

    predictions = pd.DataFrame(rows)
    summary = []
    for condition, frame in predictions.groupby("condition", sort=False):
        row = {"condition": condition}
        row.update(metrics(frame, "p_B_counterbalanced"))
        row["history_A"] = int(frame["history_A"].iloc[0]) if condition != "zero_shot" else 0
        row["history_B"] = int(frame["history_B"].iloc[0]) if condition != "zero_shot" else 0
        summary.append(row)
    summary = pd.DataFrame(summary)
    predictions.to_csv(OUTPUT_DIR / "predictions_private.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

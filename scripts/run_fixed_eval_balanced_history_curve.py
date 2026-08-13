from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "notebooks/results/fixed_eval_balanced_history"
MODEL = "deepseek-v4-flash"
PARTICIPANTS = ["P025", "P026"]
TARGET_TRIALS = [21, 22, 23, 24]
HISTORY_LENGTHS = [0, 2, 4, 6, 8, 10, 12]
DRAW_SEEDS = [4101, 4102, 4103]


def read_key() -> str:
    for line in (PROJECT / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("DEEPSEEK_API_KEY="):
            return line.partition("=")[2].strip().strip('"').strip("'")
    raise RuntimeError("DEEPSEEK_API_KEY is missing")


def logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def extract_probability(response) -> tuple[float, dict]:
    items = response.choices[0].logprobs.content[0].top_logprobs
    groups = {"A": [], "B": []}
    top = []
    for item in items:
        token = item.token
        top.append({"token": token, "logprob": float(item.logprob)})
        label = token.strip()
        if label in groups:
            groups[label].append(float(item.logprob))
    if not groups["A"] or not groups["B"]:
        raise ValueError("A and B must both appear in top_logprobs")
    label_lp = {label: logsumexp(values) for label, values in groups.items()}
    log_z = logsumexp(list(label_lp.values()))
    p_b = math.exp(label_lp["B"] - log_z)
    return p_b, {"top_logprobs": top, "surface_forms": groups, "label_logprob": label_lp}


def make_messages(target, history: pd.DataFrame) -> list[dict]:
    history_text = ""
    if len(history):
        blocks = []
        for _, row in history.sort_values("trial_index").iterrows():
            blocks.append(
                f"Earlier trial {int(row['trial_index'])}:\n"
                f"Option A: {row['lottery_A']}\n"
                f"Option B: {row['lottery_B']}\n"
                f"Observed choice: {row['choice']}"
            )
        history_text = (
            "Here are earlier choices from the same participant:\n\n"
            + "\n\n".join(blocks)
            + "\n\n"
        )
    return [
        {"role": "system", "content": "Answer with exactly one token: A or B."},
        {
            "role": "user",
            "content": (
                history_text
                + "Predict this participant's choice on a new trial.\n"
                + f"Option A: {target['lottery_A']}\n"
                + f"Option B: {target['lottery_B']}\n"
                + "Answer:"
            ),
        },
    ]


def call_with_retry(client, messages, attempts=4):
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


def score_row(response, target, history, draw):
    p_b, logprob_details = extract_probability(response)
    y = int(target["choice"] == "B")
    return {
        "participant_id": target["participant_id"],
        "target_trial": int(target["trial_index"]),
        "history_trials": int(len(history)),
        "history_A": int((history["choice"] == "A").sum()),
        "history_B": int((history["choice"] == "B").sum()),
        "history_draw": int(draw),
        "actual_choice": target["choice"],
        "sampled_output": response.choices[0].message.content,
        "p_B": p_b,
        "predicted_choice": "B" if p_b >= 0.5 else "A",
        "correct": int((p_b >= 0.5) == bool(y)),
        "log_loss": -(y * math.log(max(p_b, 1e-12)) + (1 - y) * math.log(max(1 - p_b, 1e-12))),
        "brier": (p_b - y) ** 2,
        "model": response.model,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        **logprob_details,
    }


OUT.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(PROJECT / "notebooks/data/behavioral_expanded_public_slice.csv")
pilot_rows = [json.loads(line) for line in (PROJECT / "notebooks/results/deepseek_history_curve_pilot.jsonl").read_text(encoding="utf-8").splitlines()]
pilot_zero = {
    (row["participant_id"], int(row["target_trial"])): row
    for row in pilot_rows
    if int(row["history_trials"]) == 0
}
assert set(pilot_zero) == {(p, t) for p in PARTICIPANTS for t in TARGET_TRIALS}

# One nested A-order and B-order per participant and draw. Selected trials are
# restored to chronological order before prompting; only membership changes.
orders = {}
manifest = []
for draw, seed in enumerate(DRAW_SEEDS, start=1):
    for participant_number, participant in enumerate(PARTICIPANTS):
        person = data[data["participant_id"] == participant].sort_values("trial_index")
        pool = person[person["trial_index"] <= 20]
        rng = np.random.default_rng(seed + participant_number * 100)
        orders[(draw, participant)] = {}
        for label in ["A", "B"]:
            indices = pool.loc[pool["choice"] == label].index.to_numpy()
            orders[(draw, participant)][label] = rng.permutation(indices).tolist()
            assert len(indices) >= max(HISTORY_LENGTHS) // 2
        for k in HISTORY_LENGTHS[1:]:
            half = k // 2
            selected = person.loc[
                orders[(draw, participant)]["A"][:half]
                + orders[(draw, participant)]["B"][:half]
            ].sort_values("trial_index")
            assert selected["choice"].value_counts().to_dict() == {"A": half, "B": half}
            for _, row in selected.iterrows():
                manifest.append({
                    "participant_id": participant,
                    "target_trials": "21-24",
                    "history_draw": draw,
                    "history_trials": k,
                    "trial_index": int(row["trial_index"]),
                    "choice": row["choice"],
                })
pd.DataFrame(manifest).to_csv(OUT / "selection_manifest.csv", index=False)

details_path = OUT / "fixed_eval_balanced_details.jsonl"
completed = {}
if details_path.exists():
    for line in details_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        completed[(row["participant_id"], int(row["target_trial"]), int(row["history_trials"]), int(row["history_draw"]))] = row

# Reuse the exact pilot zero-shot responses. They do not depend on a draw.
for (participant, target_trial), row in pilot_zero.items():
    copied = dict(row)
    copied.update({"history_A": 0, "history_B": 0, "history_draw": 0})
    completed[(participant, target_trial, 0, 0)] = copied

client = OpenAI(api_key=read_key(), base_url="https://api.deepseek.com")
for draw in range(1, len(DRAW_SEEDS) + 1):
    for participant in PARTICIPANTS:
        person = data[data["participant_id"] == participant].sort_values("trial_index")
        for target_trial in TARGET_TRIALS:
            target = person[person["trial_index"] == target_trial].iloc[0]
            for k in HISTORY_LENGTHS[1:]:
                key = (participant, target_trial, k, draw)
                if key in completed:
                    continue
                half = k // 2
                indices = (
                    orders[(draw, participant)]["A"][:half]
                    + orders[(draw, participant)]["B"][:half]
                )
                history = person.loc[indices].sort_values("trial_index")
                response = call_with_retry(client, make_messages(target, history))
                row = score_row(response, target, history, draw)
                with details_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed[key] = row

results = pd.DataFrame(completed.values()).sort_values(
    ["participant_id", "target_trial", "history_trials", "history_draw"]
)
results.to_csv(OUT / "fixed_eval_balanced_details.csv", index=False)

# Aggregate within each draw first, then summarize draws. k=0 is one shared
# estimate and therefore has zero between-draw uncertainty.
draw_summary = (
    results.groupby(["history_trials", "history_draw"], as_index=False)
    .agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        log_loss=("log_loss", "mean"),
        brier=("brier", "mean"),
        mean_p_B=("p_B", "mean"),
    )
)
summary = (
    draw_summary.groupby("history_trials", as_index=False)
    .agg(
        draws=("history_draw", "size"),
        n_per_draw=("n", "first"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        log_loss_mean=("log_loss", "mean"),
        log_loss_sd=("log_loss", "std"),
        brier_mean=("brier", "mean"),
        mean_p_B=("mean_p_B", "mean"),
    )
)
participant_draw = (
    results.groupby(["participant_id", "history_trials", "history_draw"], as_index=False)
    .agg(n=("correct", "size"), accuracy=("correct", "mean"), log_loss=("log_loss", "mean"))
)
participant_summary = (
    participant_draw.groupby(["participant_id", "history_trials"], as_index=False)
    .agg(
        draws=("history_draw", "size"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        log_loss_mean=("log_loss", "mean"),
        log_loss_sd=("log_loss", "std"),
    )
)
draw_summary.to_csv(OUT / "fixed_eval_balanced_by_draw.csv", index=False)
summary.to_csv(OUT / "fixed_eval_balanced_summary.csv", index=False)
participant_summary.to_csv(OUT / "fixed_eval_balanced_by_participant.csv", index=False)
print("\nSUMMARY")
print(summary.to_string(index=False))

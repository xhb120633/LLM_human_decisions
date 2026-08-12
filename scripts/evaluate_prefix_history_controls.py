from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-0.6B"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "notebooks" / "data" / "behavioral_expanded_public_slice.csv"
ADAPTER_PATH = (
    PROJECT_ROOT / "artifacts" / "sft" / "local_qwen3_0_6b_prefix_history_sft" / "adapter"
)
OUTPUT_DIR = ADAPTER_PATH.parent


def instruction(tokenizer):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Predict a participant's choices in a risky-choice session."},
            {"role": "user", "content": (
                "Use the earlier trials and observed choices from the same participant "
                "to predict the current choice. Answer with exactly A or B."
            )},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def trial_text(row):
    return (
        f"Trial {int(row['trial_index'])}:\n"
        f"Option A: {row['lottery_A']}\n"
        f"Option B: {row['lottery_B']}\n"
        "Observed choice:"
    )


def deterministic_permutation(values, participant, target_trial):
    seed_text = f"{participant}-{target_trial}-choice-shuffle".encode()
    seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return rng.permutation(np.asarray(values, dtype=object))


def prompt_ids(tokenizer, person, target_trial, condition):
    ids = encode(tokenizer, instruction(tokenizer))
    earlier = person[person["trial_index"] < target_trial].copy()
    if condition == "current_only":
        earlier = earlier.iloc[0:0]
    elif condition == "choice_shuffled":
        earlier["choice"] = deterministic_permutation(
            earlier["choice"].values,
            person["participant_id"].iloc[0],
            target_trial,
        )
    elif condition != "true_history":
        raise ValueError(condition)

    for _, row in earlier.iterrows():
        ids.extend(encode(tokenizer, trial_text(row)))
        ids.extend(encode(tokenizer, " " + str(row["choice"])))
        ids.extend(encode(tokenizer, "\n\n"))
    target = person[person["trial_index"] == target_trial].iloc[0]
    ids.extend(encode(tokenizer, trial_text(target)))
    return ids, target


@torch.inference_mode()
def score_next_ab(model, tokenizer, ids):
    a_id = encode(tokenizer, " A")
    b_id = encode(tokenizer, " B")
    assert len(a_id) == len(b_id) == 1
    sequence = torch.tensor([ids], device=model.device)
    logprobs = model(input_ids=sequence).logits[0, -1].log_softmax(-1)
    log_a = float(logprobs[a_id[0]].cpu())
    log_b = float(logprobs[b_id[0]].cpu())
    peak = max(log_a, log_b)
    z = peak + math.log(math.exp(log_a - peak) + math.exp(log_b - peak))
    return math.exp(log_b - z)


def summarize(condition, frame):
    y = (frame["choice"] == "B").astype(int)
    p_b = frame["p_B"].clip(1e-6, 1 - 1e-6)
    return {
        "condition": condition,
        "n": len(frame),
        "accuracy": accuracy_score(y, p_b >= 0.5),
        "log_loss": log_loss(y, p_b, labels=[0, 1]),
        "brier": brier_score_loss(y, p_b),
        "mean_p_B": p_b.mean(),
        "predicted_B_rate": (p_b >= 0.5).mean(),
        "observed_B_rate": y.mean(),
    }


def main():
    data = pd.read_csv(DATA_PATH)
    participants = sorted(data["participant_id"].unique())
    test_ids = participants[24:]
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH).to("cuda").eval()

    rows = []
    for condition in ["current_only", "true_history", "choice_shuffled"]:
        for participant in test_ids:
            person = data[data["participant_id"] == participant].sort_values("trial_index")
            for target_trial in range(21, 41):
                ids, target = prompt_ids(tokenizer, person, target_trial, condition)
                rows.append(
                    {
                        "condition": condition,
                        "participant_id": participant,
                        "trial_index": target_trial,
                        "choice": target["choice"],
                        "p_B": score_next_ab(model, tokenizer, ids),
                    }
                )

    predictions = pd.DataFrame(rows)
    metrics = pd.DataFrame(
        [summarize(condition, frame) for condition, frame in predictions.groupby("condition")]
    ).sort_values("condition")
    predictions.to_csv(OUTPUT_DIR / "predictions_history_controls.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics_history_controls.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


SEED = 3000
MODEL_NAME = os.getenv("SFT_MODEL_NAME", "Qwen/Qwen3-0.6B")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "notebooks" / "data" / "behavioral_expanded_public_slice.csv"
MODEL_SLUG = MODEL_NAME.split("/")[-1].lower().replace(".", "_").replace("-", "_")
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "sft" / f"local_{MODEL_SLUG}_prefix_history_sft"


def session_instruction(tokenizer):
    messages = [
        {
            "role": "system",
            "content": "Predict a participant's choices in a risky-choice session.",
        },
        {
            "role": "user",
            "content": (
                "Use the earlier trials and observed choices from the same participant "
                "to predict the current choice. Answer with exactly A or B."
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def trial_prefix(row):
    return (
        f"Trial {int(row['trial_index'])}:\n"
        f"Option A: {row['lottery_A']}\n"
        f"Option B: {row['lottery_B']}\n"
        "Observed choice:"
    )


def encode_text(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def prefix_ids_for_target(tokenizer, person, target_trial, include_history=True):
    ids = encode_text(tokenizer, session_instruction(tokenizer))
    ordered = person.sort_values("trial_index")
    if include_history:
        for _, row in ordered[ordered["trial_index"] < target_trial].iterrows():
            ids.extend(encode_text(tokenizer, trial_prefix(row)))
            ids.extend(encode_text(tokenizer, " " + str(row["choice"])))
            ids.extend(encode_text(tokenizer, "\n\n"))
    target = ordered[ordered["trial_index"] == target_trial].iloc[0]
    ids.extend(encode_text(tokenizer, trial_prefix(target)))
    return ids


class PrefixHistoryDataset(Dataset):
    """One target decision per example; only its A/B completion receives loss."""

    def __init__(self, frame, tokenizer):
        self.examples = []
        self.metadata = []
        for participant, person in frame.groupby("participant_id", sort=True):
            person = person.sort_values("trial_index")
            for target_trial in person["trial_index"].astype(int):
                prompt_ids = prefix_ids_for_target(
                    tokenizer, person, target_trial, include_history=True
                )
                choice = str(
                    person.loc[person["trial_index"] == target_trial, "choice"].iloc[0]
                )
                choice_ids = encode_text(tokenizer, " " + choice)
                if len(choice_ids) != 1:
                    raise ValueError(f"Expected one-token A/B target, got {choice_ids}")
                self.examples.append(
                    {
                        "input_ids": prompt_ids + choice_ids,
                        "attention_mask": [1] * (len(prompt_ids) + 1),
                        "labels": [-100] * len(prompt_ids) + choice_ids,
                    }
                )
                self.metadata.append(
                    {
                        "participant_id": participant,
                        "target_trial": target_trial,
                        "history_trials": target_trial - 1,
                        "tokens": len(prompt_ids) + 1,
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collator(tokenizer):
    def collate(examples):
        length = max(len(example["input_ids"]) for example in examples)
        batch = len(examples)
        input_ids = torch.full((batch, length), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, length), dtype=torch.long)
        labels = torch.full((batch, length), -100, dtype=torch.long)
        for index, example in enumerate(examples):
            size = len(example["input_ids"])
            input_ids[index, :size] = torch.tensor(example["input_ids"])
            attention_mask[index, :size] = 1
            labels[index, :size] = torch.tensor(example["labels"])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def lse(a, b):
    peak = max(a, b)
    return peak + math.log(math.exp(a - peak) + math.exp(b - peak))


@torch.inference_mode()
def ab_logprobs(model, tokenizer, prompt_ids):
    """Read A/B from one next-token distribution instead of two forward passes."""
    a_ids = encode_text(tokenizer, " A")
    b_ids = encode_text(tokenizer, " B")
    if len(a_ids) != 1 or len(b_ids) != 1:
        raise ValueError("This evaluation expects single-token ' A' and ' B' labels")
    sequence = torch.tensor([prompt_ids], device=model.device)
    next_token_logprobs = model(input_ids=sequence).logits[0, -1].log_softmax(-1)
    return float(next_token_logprobs[a_ids[0]].cpu()), float(next_token_logprobs[b_ids[0]].cpu())


def score(model, tokenizer, data, test_ids, include_history, condition):
    rows = []
    model.eval()
    for participant in test_ids:
        person = data[data["participant_id"] == participant].sort_values("trial_index")
        for target_trial in range(21, 41):
            target = person[person["trial_index"] == target_trial].iloc[0]
            prompt_ids = prefix_ids_for_target(tokenizer, person, target_trial, include_history)
            log_a, log_b = ab_logprobs(model, tokenizer, prompt_ids)
            normalizer = lse(log_a, log_b)
            rows.append(
                {
                    "condition": condition,
                    "participant_id": participant,
                    "trial_index": target_trial,
                    "history_trials": target_trial - 1 if include_history else 0,
                    "choice": target["choice"],
                    "p_B": math.exp(log_b - normalizer),
                    "logprob_A": log_a,
                    "logprob_B": log_b,
                }
            )
    return pd.DataFrame(rows)


def metric_row(condition, frame):
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    participants = sorted(data["participant_id"].unique())
    train_ids = participants[:20]
    validation_ids = participants[20:24]
    test_ids = participants[24:]
    train = data[data["participant_id"].isin(train_ids)]
    validation = data[data["participant_id"].isin(validation_ids)]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = PrefixHistoryDataset(train, tokenizer)
    validation_dataset = PrefixHistoryDataset(validation, tokenizer)
    metadata = pd.DataFrame(train_dataset.metadata)
    print(
        "Training examples:", len(train_dataset),
        "validation examples:", len(validation_dataset),
        "token range:", (metadata.tokens.min(), round(metadata.tokens.mean(), 1), metadata.tokens.max()),
    )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to("cuda")
    base.config.use_cache = False
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    # Required when activation checkpointing is combined with a frozen base
    # model: gradients must be allowed to enter through the input embeddings.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR / "trainer"),
            num_train_epochs=3,
            # Full participant histories are long. A micro-batch of one plus
            # checkpointing keeps substantial headroom on a 24 GB teaching GPU.
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            optim="adamw_torch",
            report_to="none",
            remove_unused_columns=False,
            group_by_length=True,
            gradient_checkpointing=True,
            seed=SEED,
        ),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator(tokenizer),
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR / "adapter")
    tokenizer.save_pretrained(OUTPUT_DIR / "adapter")

    rows = []
    with trainer.model.disable_adapter():
        rows.append(score(trainer.model, tokenizer, data, test_ids, False, "base_current_only"))
        rows.append(score(trainer.model, tokenizer, data, test_ids, True, "base_full_history"))
    rows.append(score(trainer.model, tokenizer, data, test_ids, False, "sft_current_only"))
    rows.append(score(trainer.model, tokenizer, data, test_ids, True, "sft_full_history"))
    predictions = pd.concat(rows, ignore_index=True)
    metrics = pd.DataFrame(
        [metric_row(condition, frame) for condition, frame in predictions.groupby("condition")]
    ).sort_values("condition")

    predictions.to_csv(OUTPUT_DIR / "predictions_2x2_full_history.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics_2x2_full_history.csv", index=False)
    metadata.to_csv(OUTPUT_DIR / "training_example_metadata.csv", index=False)
    (OUTPUT_DIR / "training_summary.json").write_text(
        json.dumps(
            {
                "train_metrics": result.metrics,
                "log_history": trainer.state.log_history,
                "train_examples": len(train_dataset),
                "validation_examples": len(validation_dataset),
                "optimizer_steps": trainer.state.global_step,
                "train_participants": train_ids,
                "validation_participants": validation_ids,
                "test_participants": test_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nMETRICS")
    print(metrics.to_string(index=False))
    print("Saved to", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()

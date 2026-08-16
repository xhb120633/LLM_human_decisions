"""Recover masked A/B choices with Qwen and save renormalized label probabilities.

This is the portable generation stage used by Notebook 2. It performs no free
text generation: one forward pass yields next-token logits, which are
renormalized over the valid labels A and B and written to an NPZ file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gpu-memory", default="20GiB")
    parser.add_argument("--cpu-memory", default="32GiB")
    return parser.parse_args()


def load_records(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        sentences = pd.read_csv(path)
        required = {"trial_row", "sentence_index", "sentence_text", "question", "choice"}
        missing = required.difference(sentences.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        return (
            sentences.sort_values(["trial_row", "sentence_index"])
            .groupby("trial_row", as_index=False)
            .agg(
                question=("question", "first"),
                reasoning=("sentence_text", " ".join),
                choice=("choice", "first"),
            )
        )
    array = np.load(path, allow_pickle=True)
    return pd.DataFrame(
        {
            "question": array[:, 10].astype(str),
            "reasoning": array[:, 8].astype(str),
            "choice": array[:, 2].astype(int),
        }
    )


def make_prompt(tokenizer, record) -> str:
    question = str(record.question)
    reasoning = str(record.reasoning)
    messages = [
        {
            "role": "system",
            "content": (
                "You recover a masked choice from a reasoning trace. "
                "Output exactly one label: A or B."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{question}\n\n"
                "The final choice in this synthetic participant reasoning was "
                'replaced by "Option X". Recover X from the reasoning; do not '
                "make your own decision.\n\n"
                f"Masked reasoning:\n{reasoning}\n\n"
                "Which option did this participant choose?"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def continuation_token_id(tokenizer, prompt: str, label: str) -> int:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completed_ids = tokenizer(prompt + label, add_special_tokens=False).input_ids
    suffix = completed_ids[len(prompt_ids) :]
    if len(suffix) != 1:
        raise ValueError(f"{label!r} is not one token after the prompt: {suffix}")
    return int(suffix[0])


def metrics(labels: np.ndarray, probability_b: np.ndarray) -> dict[str, float | int]:
    predicted = (probability_b >= 0.5).astype(np.int64)
    recall_a = float((predicted[labels == 0] == 0).mean())
    recall_b = float((predicted[labels == 1] == 1).mean())
    clipped = np.clip(probability_b, 1e-12, 1 - 1e-12)
    return {
        "n": int(len(labels)),
        "accuracy": float((predicted == labels).mean()),
        "balanced_accuracy": (recall_a + recall_b) / 2,
        "log_loss": float(
            -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean()
        ),
        "mean_p_b": float(probability_b.mean()),
    }


def main() -> None:
    args = arguments()
    import torch
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    data = load_records(args.data_path)
    if args.max_trials:
        data = data.iloc[: args.max_trials].copy()
    labels = data["choice"].to_numpy(dtype=np.int64)
    prompts = []

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [make_prompt(tokenizer, record) for record in data.itertuples(index=False)]
    a_id = continuation_token_id(tokenizer, prompts[0], "A")
    b_id = continuation_token_id(tokenizer, prompts[0], "B")

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        attn_implementation="sdpa",
    ).eval()
    candidate_ids = torch.tensor([a_id, b_id], device=model.device)
    probability_b = np.full(len(prompts), np.nan, dtype=np.float64)

    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(model.device)
        with torch.inference_mode():
            output = model(**inputs, use_cache=False)
            label_logits = output.logits[:, -1, candidate_ids].float()
            probability_b[start : start + len(batch)] = (
                label_logits.softmax(dim=-1)[:, 1].cpu().numpy()
            )
        print(f"processed {min(start + len(batch), len(prompts))}/{len(prompts)}", flush=True)

    if not np.isfinite(probability_b).all():
        raise RuntimeError("Some trials did not receive a probability")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_path, p_b=probability_b, labels=labels)
    report = metrics(labels, probability_b)
    args.output_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

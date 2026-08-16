"""Extract sentence-end hidden states from Qwen3.5, one memory-mapped file per layer.

The model reads the full masked reasoning trace once. Because Qwen3.5 is
autoregressive, the hidden state at a sentence-end token can only depend on
tokens at or before that position. Row i in every layer_XX.npy corresponds to
row i in sentences.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions-path", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--package-path",
        type=Path,
        help="Optional isolated site-packages directory containing a Qwen3.5-compatible Transformers.",
    )
    parser.add_argument("--gpu-memory", default="17GiB")
    parser.add_argument("--cpu-memory", default="32GiB")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--verify-causality", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def add_package_path(path: Path | None) -> None:
    if path is not None:
        sys.path.insert(0, str(path.resolve()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Return trimmed sentence character spans without changing the source text."""
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for boundary in re.finditer(r"(?<=[.!?])\s+", text):
        raw_start, raw_end = cursor, boundary.start()
        segment = text[raw_start:raw_end]
        left = len(segment) - len(segment.lstrip())
        right = len(segment.rstrip())
        if right > left:
            start, end = raw_start + left, raw_start + right
            spans.append((start, end, text[start:end]))
        cursor = boundary.end()
    segment = text[cursor:]
    left = len(segment) - len(segment.lstrip())
    right = len(segment.rstrip())
    if right > left:
        start, end = cursor + left, cursor + right
        spans.append((start, end, text[start:end]))
    return spans


def make_prompt(tokenizer: Any, question: str, reasoning: str) -> str:
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


@dataclass
class TrialEncoding:
    prompt: str
    sentence_spans: list[tuple[int, int, str]]
    sentence_end_tokens: list[int]
    input_ids: list[int]


def encode_trial(tokenizer: Any, question: str, reasoning: str) -> TrialEncoding:
    prompt = make_prompt(tokenizer, question, reasoning)
    reasoning_start = prompt.index(reasoning)
    spans = sentence_spans(reasoning)
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    sentence_end_tokens: list[int] = []
    for _, sentence_end, _ in spans:
        absolute_end = reasoning_start + sentence_end
        candidates = [
            token_index
            for token_index, (token_start, token_end) in enumerate(offsets)
            if token_end > reasoning_start
            and token_end <= absolute_end
            and token_end > token_start
        ]
        if not candidates:
            raise RuntimeError(f"Could not map sentence ending at character {absolute_end}")
        sentence_end_tokens.append(candidates[-1])
    return TrialEncoding(
        prompt=prompt,
        sentence_spans=spans,
        sentence_end_tokens=sentence_end_tokens,
        input_ids=encoded["input_ids"],
    )


def configure_process(torch_threads: int) -> None:
    import torch

    torch.set_num_threads(torch_threads)
    if os.name == "nt":
        try:
            import psutil

            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            pass


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_source_records(path: Path) -> Any:
    """Load the original NPY format or reconstruct it from the public sentence CSV."""
    import numpy as np
    import pandas as pd

    if path.suffix.lower() != ".csv":
        return np.load(path, allow_pickle=True)

    sentences = pd.read_csv(path)
    required = {
        "trial_row",
        "participant_id",
        "problem_id",
        "choice",
        "persona",
        "sentence_index",
        "sentence_text",
        "question",
    }
    missing = required.difference(sentences.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    rows = []
    grouped = sentences.sort_values(["trial_row", "sentence_index"]).groupby(
        "trial_row", sort=True
    )
    for trial_row, trial in grouped:
        record = np.empty(12, dtype=object)
        record[:] = None
        record[0] = trial["problem_id"].iloc[0]
        record[1] = trial["participant_id"].iloc[0]
        record[2] = int(trial["choice"].iloc[0])
        record[7] = int(trial_row)
        record[8] = " ".join(trial["sentence_text"].astype(str))
        record[10] = trial["question"].iloc[0]
        record[11] = trial["persona"].iloc[0]
        rows.append(record)
    return np.asarray(rows, dtype=object)


def open_layer_arrays(
    output_dir: Path,
    layer_count: int,
    sentence_count: int,
    hidden_size: int,
    storage_dtype: str,
    resume: bool,
) -> list[Any]:
    import numpy as np

    arrays = []
    mode = "r+" if resume else "w+"
    for layer_index in range(layer_count):
        path = output_dir / f"layer_{layer_index:02d}.npy"
        if resume and not path.exists():
            raise FileNotFoundError(f"Missing layer file required for resume: {path}")
        array = np.lib.format.open_memmap(
            path,
            mode=mode,
            dtype=storage_dtype,
            shape=(sentence_count, hidden_size),
        )
        arrays.append(array)
    return arrays


def write_metadata(
    path: Path,
    data: Any,
    encodings: list[TrialEncoding],
    predictions: Any | None,
) -> list[tuple[int, int]]:
    row_ranges: list[tuple[int, int]] = []
    sentence_row = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sentence_row",
                "trial_row",
                "participant_id",
                "problem_id",
                "choice",
                "persona",
                "sentence_index",
                "sentence_text",
                "reasoning_char_start",
                "reasoning_char_end",
                "prompt_token_end",
                "model_p_b",
            ],
        )
        writer.writeheader()
        for trial_row, (record, encoding) in enumerate(zip(data, encodings)):
            start_row = sentence_row
            for sentence_index, ((char_start, char_end, text), token_end) in enumerate(
                zip(encoding.sentence_spans, encoding.sentence_end_tokens)
            ):
                writer.writerow(
                    {
                        "sentence_row": sentence_row,
                        "trial_row": trial_row,
                        "participant_id": record[1],
                        "problem_id": record[0],
                        "choice": int(record[2]),
                        "persona": record[11],
                        "sentence_index": sentence_index,
                        "sentence_text": text,
                        "reasoning_char_start": char_start,
                        "reasoning_char_end": char_end,
                        "prompt_token_end": token_end,
                        "model_p_b": (
                            "" if predictions is None else float(predictions[trial_row])
                        ),
                    }
                )
                sentence_row += 1
            row_ranges.append((start_row, sentence_row))
    return row_ranges


def extract_hidden_states(output: Any) -> tuple[Any, ...]:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        language_output = getattr(output, "language_model_output", None)
        hidden_states = getattr(language_output, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model output did not include hidden_states")
    return tuple(hidden_states)


def verify_causal_equivalence(
    model: Any,
    tokenizer: Any,
    record: Any,
    full_encoding: TrialEncoding,
    full_hidden_states: tuple[Any, ...],
    device: str,
) -> dict[str, float]:
    import torch

    sentence_index = max(0, len(full_encoding.sentence_spans) // 2 - 1)
    _, prefix_end, _ = full_encoding.sentence_spans[sentence_index]
    reasoning_prefix = str(record[8])[:prefix_end]
    prefix_encoding = encode_trial(tokenizer, str(record[10]), reasoning_prefix)
    full_token_index = full_encoding.sentence_end_tokens[sentence_index]
    prefix_token_index = prefix_encoding.sentence_end_tokens[-1]

    full_prefix_ids = full_encoding.input_ids[: full_token_index + 1]
    truncated_prefix_ids = prefix_encoding.input_ids[: prefix_token_index + 1]
    if full_prefix_ids != truncated_prefix_ids:
        raise RuntimeError("Full and truncated prompts do not share identical prefix tokens")

    inputs = tokenizer(
        prefix_encoding.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)
    with torch.inference_mode():
        prefix_output = model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    prefix_hidden_states = extract_hidden_states(prefix_output)
    max_absolute_difference = 0.0
    minimum_cosine_similarity = 1.0
    for full_layer, prefix_layer in zip(full_hidden_states, prefix_hidden_states):
        full_vector = full_layer[0, full_token_index].float()
        prefix_vector = prefix_layer[0, prefix_token_index].float()
        difference = float((full_vector - prefix_vector).abs().max().cpu())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                full_vector.unsqueeze(0), prefix_vector.unsqueeze(0)
            ).cpu()
        )
        max_absolute_difference = max(max_absolute_difference, difference)
        minimum_cosine_similarity = min(minimum_cosine_similarity, cosine)
    return {
        "trial_row": int(record[7][0]) if hasattr(record[7], "__len__") else int(record[7]),
        "sentence_index": sentence_index,
        "max_absolute_difference": max_absolute_difference,
        "minimum_cosine_similarity": minimum_cosine_similarity,
    }


def main() -> None:
    args = parse_args()
    add_package_path(args.package_path)
    configure_process(args.torch_threads)

    import numpy as np
    import torch
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_source_records(args.data_path)
    if args.max_trials:
        data = data[: args.max_trials]

    predictions = None
    if args.predictions_path is not None:
        predictions_file = np.load(args.predictions_path)
        predictions = predictions_file["p_b"][: len(data)]
        if not np.isfinite(predictions).all():
            raise ValueError("Predictions file contains unfinished rows")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encodings = [
        encode_trial(tokenizer, str(record[10]), str(record[8])) for record in data
    ]
    sentence_count = sum(len(item.sentence_spans) for item in encodings)
    metadata_path = args.output_dir / "sentences.csv"
    row_ranges = write_metadata(metadata_path, data, encodings, predictions)

    progress_path = args.output_dir / "progress.json"
    manifest_path = args.output_dir / "manifest.json"
    progress = load_json(progress_path, {"completed_trials": 0, "complete": False})
    resume = progress["completed_trials"] > 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        offload_folder=str(args.output_dir / "offload"),
        attn_implementation="sdpa",
    ).eval()
    hidden_size = int(model.config.text_config.hidden_size)
    layer_count = int(model.config.text_config.num_hidden_layers) + 1

    manifest = {
        "format_version": 1,
        "model": args.model,
        "source_file": args.data_path.name,
        "source_sha256": sha256_file(args.data_path),
        "source_dtype": "bfloat16",
        "storage_dtype": args.storage_dtype,
        "trial_count": len(data),
        "sentence_count": sentence_count,
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "row_alignment": "Row i in every layer file corresponds to row i in sentences.csv",
        "layer_semantics": {
            "layer_00": "input token embedding / initial residual state",
            f"layer_{layer_count - 1:02d}": "final transformer block output",
        },
        "prompt_condition": "choice-recovery task conditioned",
        "causal_interpretation": (
            "Each sentence-end state encodes only the prompt prefix through that token."
        ),
    }
    save_json(manifest_path, manifest)

    layer_arrays = open_layer_arrays(
        args.output_dir,
        layer_count,
        sentence_count,
        hidden_size,
        args.storage_dtype,
        resume,
    )
    verification_results: list[dict[str, float]] = []
    start_trial = int(progress["completed_trials"])
    start_time = time.time()

    for trial_row in range(start_trial, len(data)):
        encoding = encodings[trial_row]
        inputs = tokenizer(
            encoding.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to("cuda")
        with torch.inference_mode():
            output = model(
                **inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = extract_hidden_states(output)
        if len(hidden_states) != layer_count:
            raise RuntimeError(
                f"Expected {layer_count} hidden-state tensors; received {len(hidden_states)}"
            )
        row_start, row_end = row_ranges[trial_row]
        token_indices = torch.tensor(
            encoding.sentence_end_tokens,
            device=hidden_states[0].device,
            dtype=torch.long,
        )
        for layer_index, layer_hidden in enumerate(hidden_states):
            values = (
                layer_hidden[0]
                .index_select(0, token_indices)
                .to(dtype=torch.float16 if args.storage_dtype == "float16" else torch.float32)
                .cpu()
                .numpy()
            )
            layer_arrays[layer_index][row_start:row_end] = values

        if trial_row < args.verify_causality:
            verification_results.append(
                verify_causal_equivalence(
                    model,
                    tokenizer,
                    data[trial_row],
                    encoding,
                    hidden_states,
                    "cuda",
                )
            )

        completed_trials = trial_row + 1
        if (
            completed_trials % args.checkpoint_every == 0
            or completed_trials == len(data)
        ):
            for array in layer_arrays:
                array.flush()
            save_json(
                progress_path,
                {
                    "completed_trials": completed_trials,
                    "complete": completed_trials == len(data),
                },
            )
            print(
                f"processed {completed_trials}/{len(data)} trials; "
                f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                flush=True,
            )
        time.sleep(args.sleep_seconds)

    save_json(
        args.output_dir / "causal_verification.json",
        {"checks": verification_results},
    )
    manifest["elapsed_seconds"] = time.time() - start_time
    manifest["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
    manifest["complete"] = True
    save_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

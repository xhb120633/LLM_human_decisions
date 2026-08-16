"""Efficiently extract Qwen3.5-9B layer-15 states for Choice13K stimuli.

Only the first 15 transformer blocks are evaluated. A forward hook captures
the output of block 15 before the model's final normalization, matching
``output_hidden_states=True`` layer index 15 from a complete forward pass.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from extract_qwen35_sentence_states import (
    add_package_path,
    configure_process,
    save_json,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--package-path", type=Path)
    parser.add_argument("--gpu-memory", default="14GiB")
    parser.add_argument("--cpu-memory", default="48GiB")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-stimuli", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported decoder-layer output type: {type(output)}")


def main() -> None:
    args = arguments()
    add_package_path(args.package_path)
    configure_process(args.torch_threads)

    import numpy as np
    import pandas as pd
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stimuli = pd.read_csv(args.stimuli_path)
    if args.max_stimuli:
        stimuli = stimuli.iloc[: args.max_stimuli].copy()

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    text_model = model.model.language_model
    full_layer_count = len(text_model.layers)
    if not 1 <= args.layer <= full_layer_count:
        raise ValueError(f"Layer must be in [1, {full_layer_count}]")
    hidden_size = int(text_model.config.hidden_size)

    # Hidden-state index 15 is the output of transformer block 15 (zero-based
    # module index 14). Stop the text model immediately after that block.
    captured: dict[str, torch.Tensor] = {}

    def capture_layer(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        captured["state"] = tensor_output(output)

    hook = text_model.layers[args.layer - 1].register_forward_hook(capture_layer)
    text_model.config.num_hidden_layers = args.layer

    state_path = args.output_dir / f"stimulus_layer_{args.layer:02d}.npy"
    progress_path = args.output_dir / "progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {"completed_stimuli": 0, "complete": False}
    )
    resume = int(progress["completed_stimuli"]) > 0
    states = np.lib.format.open_memmap(
        state_path,
        mode="r+" if resume else "w+",
        dtype=np.float16,
        shape=(len(stimuli), hidden_size),
    )

    started = time.time()
    start = int(progress["completed_stimuli"])
    for batch_number, batch_start in enumerate(
        range(start, len(stimuli), args.batch_size), start=1
    ):
        batch_end = min(batch_start + args.batch_size, len(stimuli))
        texts = stimuli.iloc[batch_start:batch_end]["stimulus_text"].tolist()
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        ).to("cuda")
        captured.clear()
        with torch.inference_mode():
            text_output = text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        if "state" not in captured:
            raise RuntimeError("Layer hook did not capture a hidden state")
        layer_state = captured["state"]
        final_positions = inputs["attention_mask"].sum(dim=1) - 1
        batch_rows = torch.arange(len(texts), device=layer_state.device)
        values = (
            layer_state[batch_rows, final_positions]
            .to(torch.float16)
            .cpu()
            .numpy()
        )
        states[batch_start:batch_end] = values
        del text_output, layer_state, inputs, values

        if (
            batch_number % args.checkpoint_every == 0
            or batch_end == len(stimuli)
        ):
            states.flush()
            save_json(
                progress_path,
                {
                    "completed_stimuli": batch_end,
                    "complete": batch_end == len(stimuli),
                },
            )
            print(
                f"processed {batch_end}/{len(stimuli)} stimuli; "
                f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                flush=True,
            )

    hook.remove()
    manifest = {
        "format_version": 1,
        "model": args.model,
        "layer": args.layer,
        "hidden_size": hidden_size,
        "stimulus_count": len(stimuli),
        "storage_dtype": "float16",
        "state_definition": (
            "output of transformer block 15 at the final non-padding stimulus token"
        ),
        "prompt_condition": "raw stimulus text; no chat template and no task instruction",
        "computation": "early exit after transformer block 15",
        "batch_size": args.batch_size,
        "gpu_memory_limit": args.gpu_memory,
        "elapsed_seconds": time.time() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "complete": True,
    }
    save_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

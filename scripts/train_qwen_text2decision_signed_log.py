"""Train multiscale Text2Decision with signed-log monetary targets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from text2decision_transforms import (
    inverse_signed_log_monetary,
    signed_log_monetary,
)
from train_qwen_text2decision import TextDecisionModel, grouped_split


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-path", type=Path, required=True)
    parser.add_argument("--targets-path", type=Path, required=True)
    parser.add_argument("--stimuli-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def raw_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    return {
        "mse": float(mean_squared_error(target, prediction)),
        "mae": float(mean_absolute_error(target, prediction)),
        "mean_r2": float(
            r2_score(target, prediction, multioutput="uniform_average")
        ),
        "r2_by_dimension": r2_score(
            target, prediction, multioutput="raw_values"
        ).tolist(),
    }


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    x = np.asarray(np.load(args.states_path, mmap_mode="r"), dtype=np.float32)
    y_raw = np.load(args.targets_path).astype(np.float32)
    y_transformed = signed_log_monetary(y_raw)
    stimuli = pd.read_csv(args.stimuli_path)
    if len(x) != len(y_raw) or len(x) != len(stimuli):
        raise ValueError("States, targets, and metadata must have identical rows")

    splits = grouped_split(stimuli["problem_id"].to_numpy(), args.seed)
    target_mean = y_transformed[splits["train"]].mean(axis=0)
    target_std = y_transformed[splits["train"]].std(axis=0)
    y_training = (y_transformed - target_mean) / target_std
    split_rows = [
        {"stimulus_row": int(row), "split": name}
        for name, rows in splits.items()
        for row in rows
    ]
    pd.DataFrame(split_rows).sort_values("stimulus_row").to_csv(
        args.output_dir / "split_assignments.csv", index=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {}
    for name, rows in splits.items():
        loaders[name] = DataLoader(
            TensorDataset(
                torch.from_numpy(x[rows]),
                torch.from_numpy(y_training[rows]),
            ),
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    model = TextDecisionModel(x.shape[1], y_raw.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    best_validation = float("inf")
    best_epoch = 0
    patience_left = args.patience
    history = []
    checkpoint_path = args.output_dir / "TextDecisionModel_qwen_layer15.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for batch_x, batch_y in loaders["train"]:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * len(batch_x)
            train_count += len(batch_x)

        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.inference_mode():
            for batch_x, batch_y in loaders["validation"]:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                loss = criterion(model(batch_x), batch_y)
                validation_sum += float(loss.item()) * len(batch_x)
                validation_count += len(batch_x)
        train_loss = train_sum / train_count
        validation_loss = validation_sum / validation_count
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "text_dim": x.shape[1],
                    "decision_dim": y_raw.shape[1],
                    "epoch": epoch,
                    "validation_mse": validation_loss,
                    "target_mean": torch.from_numpy(target_mean),
                    "target_std": torch.from_numpy(target_std),
                    "target_transform": "signed_log1p_dollars",
                },
                checkpoint_path,
            )
        else:
            patience_left -= 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:03d}: train MSE={train_loss:.6f}, "
                f"validation MSE={validation_loss:.6f}",
                flush=True,
            )
        if patience_left == 0:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(
        args.output_dir / "training_history.csv", index=False
    )
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    report: dict[str, object] = {
        "model": "TextDecisionModel",
        "target_transform": "signed log1p of dollar-valued dimensions",
        "split_unit": "Choice13K problem_id across options and scale variants",
        "split_counts": {name: int(len(rows)) for name, rows in splits.items()},
        "best_epoch": best_epoch,
        "best_validation_mse_transformed_z": best_validation,
        "metrics_raw_12d": {},
    }
    with torch.inference_mode():
        for name, rows in splits.items():
            prediction_z = model(
                torch.from_numpy(x[rows]).to(device)
            ).cpu().numpy()
            prediction_transformed = prediction_z * target_std + target_mean
            prediction_raw = inverse_signed_log_monetary(
                prediction_transformed
            )
            report["metrics_raw_12d"][name] = raw_metrics(
                y_raw[rows], prediction_raw
            )
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

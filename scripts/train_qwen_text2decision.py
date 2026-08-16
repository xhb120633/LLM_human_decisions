"""Train Text2Decision from Qwen layer-15 stimulus states to 12D targets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TextDecisionModel(nn.Module):
    """Original Text2Decision MLP with a 4096-dimensional input."""

    def __init__(self, text_dim: int = 4096, decision_dim: int = 12):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, decision_dim),
        )

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        return self.text_proj(text)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-path", type=Path, required=True)
    parser.add_argument("--targets-path", type=Path, required=True)
    parser.add_argument("--stimuli-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def grouped_split(groups: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    all_rows = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_rows, held_rows = next(outer.split(all_rows, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    val_local, test_local = next(
        inner.split(held_rows, groups=groups[held_rows])
    )
    return {
        "train": train_rows,
        "validation": held_rows[val_local],
        "test": held_rows[test_local],
    }


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    return {
        "mse": float(mean_squared_error(target, prediction)),
        "mae": float(mean_absolute_error(target, prediction)),
        "mean_r2": float(r2_score(target, prediction, multioutput="uniform_average")),
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
    y = np.load(args.targets_path).astype(np.float32)
    stimuli = pd.read_csv(args.stimuli_path)
    if len(x) != len(y) or len(x) != len(stimuli):
        raise ValueError("States, targets, and metadata must have identical rows")

    splits = grouped_split(stimuli["problem_id"].to_numpy(), args.seed)
    target_mean = y[splits["train"]].mean(axis=0)
    target_std = y[splits["train"]].std(axis=0)
    if np.any(target_std == 0):
        raise ValueError("A target dimension has zero training variance")
    y_training = (y - target_mean) / target_std
    split_rows = []
    for name, rows in splits.items():
        for row in rows:
            split_rows.append({"stimulus_row": int(row), "split": name})
    pd.DataFrame(split_rows).sort_values("stimulus_row").to_csv(
        args.output_dir / "split_assignments.csv", index=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {}
    for name, rows in splits.items():
        dataset = TensorDataset(
            torch.from_numpy(x[rows]),
            torch.from_numpy(y_training[rows]),
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    model = TextDecisionModel(text_dim=x.shape[1], decision_dim=y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    history = []
    best_validation = float("inf")
    best_epoch = 0
    patience_left = args.patience
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
                    "decision_dim": y.shape[1],
                    "epoch": epoch,
                    "validation_mse": validation_loss,
                    "target_mean": torch.from_numpy(target_mean),
                    "target_std": torch.from_numpy(target_std),
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

    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    report: dict[str, object] = {
        "model": "TextDecisionModel",
        "input": "Qwen3.5-9B layer-15 final stimulus-token state",
        "input_dim": int(x.shape[1]),
        "target_dim": int(y.shape[1]),
        "training_target": "standardized per dimension using train split only",
        "split_unit": "Choice13K problem_id; both options stay in the same split",
        "split_counts": {name: int(len(rows)) for name, rows in splits.items()},
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation,
        "metrics": {},
    }
    with torch.inference_mode():
        for name, rows in splits.items():
            prediction_z = model(torch.from_numpy(x[rows]).to(device)).cpu().numpy()
            prediction = prediction_z * target_std + target_mean
            report["metrics"][name] = metrics(y[rows], prediction)

    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


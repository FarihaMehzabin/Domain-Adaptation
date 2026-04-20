"""NIH source-only training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from paper_v1.evaluation.flops import linear_layer_flops
from paper_v1.evaluation.params import count_parameters
from paper_v1.evaluation.reporting import write_stage_metrics
from paper_v1.models.linear_head import LinearHead
from paper_v1.training.common import build_dataloader, evaluate_model, move_batch_to_device, save_prediction_artifact
from paper_v1.training.early_stopping import EarlyStopping
from paper_v1.training.losses import masked_bce_with_logits
from paper_v1.training.regularizers import compute_fisher_diagonal
from paper_v1.utils.io import write_json


def train_stage0(
    *,
    train_dataset,
    val_dataset,
    eval_datasets: dict[str, Any],
    feature_dim: int,
    num_labels: int,
    output_dir: str | Path,
    training_config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = LinearHead(feature_dim, num_labels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("lr", 1.0e-3)),
        weight_decay=float(training_config.get("weight_decay", 1.0e-4)),
    )
    train_loader = build_dataloader(train_dataset, int(training_config.get("batch_size", 256)), shuffle=True, seed=seed)
    patience = int(training_config.get("patience", 5))
    early_stopping = EarlyStopping(patience=patience, maximize=True)
    best_score = None
    best_path = checkpoint_dir / "stage0_best.pt"
    history = []

    for epoch in range(int(training_config.get("epochs", 20))):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            moved = move_batch_to_device(batch, device)
            logits = model(moved["embedding"])
            loss = masked_bce_with_logits(logits, moved["target"], mask=moved["mask"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch["sample_id"])
            total_examples += len(batch["sample_id"])

        val_metrics = evaluate_model(
            model,
            val_dataset,
            batch_size=int(training_config.get("batch_size", 256)),
            device=device,
        )
        score = val_metrics.get("macro_auroc")
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_examples, 1),
                "val_macro_auroc": score,
                "val_macro_auprc": val_metrics.get("macro_auprc"),
            }
        )
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_dim": feature_dim,
                    "num_labels": num_labels,
                    "best_score": best_score,
                    "history": history,
                },
                best_path,
            )
        if early_stopping.step(score):
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    param_counts = count_parameters(model)
    flops = {
        "stage0_linear_head_forward_flops": linear_layer_flops(feature_dim, num_labels, bias=True),
    }
    metrics_payload: dict[str, Any] = {
        "history": history,
        "best_val_macro_auroc": best_score,
        "parameter_counts": param_counts,
        "flops": flops,
    }
    for alias, dataset in eval_datasets.items():
        metrics = evaluate_model(
            model,
            dataset,
            batch_size=int(training_config.get("batch_size", 256)),
            device=device,
        )
        prediction_path = save_prediction_artifact(output_dir / "metrics" / f"{alias}_predictions.npz", metrics["predictions"])
        metrics_without_predictions = dict(metrics)
        metrics_without_predictions["prediction_artifact"] = str(prediction_path)
        del metrics_without_predictions["predictions"]
        write_stage_metrics(output_dir / "metrics", f"{alias}_metrics.json", metrics_without_predictions)
        metrics_payload[alias] = metrics_without_predictions

    fisher = compute_fisher_diagonal(
        model,
        build_dataloader(train_dataset, int(training_config.get("batch_size", 256)), shuffle=False, seed=seed),
        device=device,
        max_batches=int(training_config.get("fisher_max_batches", 25)),
    )
    fisher_path = checkpoint_dir / "stage0_fisher.pt"
    torch.save(fisher, fisher_path)
    write_json(
        output_dir / "artifacts" / "stage0_summary.json",
        {
            "checkpoint_path": str(best_path),
            "fisher_path": str(fisher_path),
            "best_val_macro_auroc": best_score,
            "parameter_counts": param_counts,
            "flops": flops,
        },
    )
    return {
        "model": model,
        "checkpoint_path": best_path,
        "fisher_path": fisher_path,
        "history": history,
        "metrics": metrics_payload,
        "parameter_counts": param_counts,
        "flops": flops,
        "best_val_macro_auroc": best_score,
    }

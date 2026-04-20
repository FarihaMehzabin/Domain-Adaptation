"""Sequential adaptation training and baseline helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from paper_v1.evaluation.flops import linear_layer_flops, mlp_flops, retrieval_flops
from paper_v1.evaluation.params import count_parameters
from paper_v1.evaluation.reporting import write_stage_metrics
from paper_v1.models.full_model import ContinualAdaptationModel
from paper_v1.models.labelwise_trust_region import (
    LabelWiseTrustRegionAdaptationModel,
    TopKLabelWiseTrustRegionCorrectionModel,
)
from paper_v1.models.linear_head import LinearHead
from paper_v1.models.prototype_memory import PrototypeBank, PrototypeMemoryModule
from paper_v1.models.tiny_logit_correction import TinyLogitCorrection
from paper_v1.training.common import build_dataloader, evaluate_model, move_batch_to_device, save_prediction_artifact
from paper_v1.training.early_stopping import EarlyStopping
from paper_v1.training.losses import (
    gate_sparsity_loss,
    masked_bce_with_logits,
    old_prototype_margin_preservation_loss,
    residual_trust_region_loss,
    sigmoid_distillation_loss,
)
from paper_v1.training.regularizers import ewc_penalty, l2_anchor_penalty, snapshot_parameters
from paper_v1.utils.io import write_csv, write_json


def load_linear_head_checkpoint(checkpoint_path: str | Path, *, device: torch.device) -> LinearHead:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LinearHead(checkpoint["feature_dim"], checkpoint["num_labels"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


class LogitOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module, *, feature_dim: int, num_labels: int) -> None:
        super().__init__()
        self.model = model
        self.feature_dim = int(feature_dim)
        self.num_labels = int(num_labels)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        outputs = self.model(embeddings)
        if isinstance(outputs, dict):
            return outputs["logits"]
        return outputs


def _infer_model_dims(model: nn.Module) -> tuple[int, int]:
    if hasattr(model, "feature_dim") and hasattr(model, "num_labels"):
        return int(getattr(model, "feature_dim")), int(getattr(model, "num_labels"))
    if hasattr(model, "previous_model"):
        return _infer_model_dims(getattr(model, "previous_model"))
    raise AttributeError("unable to infer feature_dim and num_labels from previous model")


def build_continual_model_from_previous_model(
    *,
    previous_model: nn.Module,
    old_bank: PrototypeBank,
    training_config: dict[str, Any],
    device: torch.device,
) -> ContinualAdaptationModel:
    feature_dim, num_labels = _infer_model_dims(previous_model)
    wrapped_previous = LogitOnlyWrapper(previous_model, feature_dim=feature_dim, num_labels=num_labels)
    memory_module = old_bank.to_module(
        top_k=int(training_config.get("top_k", 8)),
        temperature=float(training_config.get("temperature", 0.1)),
    ).to(device)
    return ContinualAdaptationModel(
        wrapped_previous,
        memory_module,
        feature_dim=feature_dim,
        num_labels=num_labels,
        gate_hidden_dim=int(training_config.get("gate_hidden_dim", 64)),
        bottleneck_dim=int(training_config.get("bottleneck_dim", 128)),
        dropout=float(training_config.get("dropout", 0.1)),
        old_like_similarity_threshold=(
            float(training_config["old_like_similarity_threshold"])
            if training_config.get("old_like_similarity_threshold") is not None
            else None
        ),
        old_like_gate_cap=(
            float(training_config["old_like_gate_cap"])
            if training_config.get("old_like_gate_cap") is not None
            else None
        ),
    ).to(device)


def build_continual_model(
    *,
    previous_checkpoint_path: str | Path,
    old_bank: PrototypeBank,
    training_config: dict[str, Any],
    device: torch.device,
) -> ContinualAdaptationModel:
    previous_model = load_linear_head_checkpoint(previous_checkpoint_path, device=device)
    memory_module = old_bank.to_module(
        top_k=int(training_config.get("top_k", 8)),
        temperature=float(training_config.get("temperature", 0.1)),
    ).to(device)
    return ContinualAdaptationModel(
        previous_model,
        memory_module,
        feature_dim=previous_model.feature_dim,
        num_labels=previous_model.num_labels,
        gate_hidden_dim=int(training_config.get("gate_hidden_dim", 64)),
        bottleneck_dim=int(training_config.get("bottleneck_dim", 128)),
        dropout=float(training_config.get("dropout", 0.1)),
        old_like_similarity_threshold=(
            float(training_config["old_like_similarity_threshold"])
            if training_config.get("old_like_similarity_threshold") is not None
            else None
        ),
        old_like_gate_cap=(
            float(training_config["old_like_gate_cap"])
            if training_config.get("old_like_gate_cap") is not None
            else None
        ),
    ).to(device)


def build_labelwise_trust_region_model(
    *,
    previous_checkpoint_path: str | Path,
    old_bank: PrototypeBank,
    training_config: dict[str, Any],
    device: torch.device,
) -> LabelWiseTrustRegionAdaptationModel:
    previous_model = load_linear_head_checkpoint(previous_checkpoint_path, device=device)
    memory_module = old_bank.to_module(
        top_k=int(training_config.get("top_k", 8)),
        temperature=float(training_config.get("temperature", 0.1)),
    ).to(device)
    return LabelWiseTrustRegionAdaptationModel(
        previous_model,
        memory_module,
        feature_dim=previous_model.feature_dim,
        num_labels=previous_model.num_labels,
        gate_hidden_dim=int(training_config.get("gate_hidden_dim", 16)),
        bottleneck_dim=int(training_config.get("bottleneck_dim", 128)),
        dropout=float(training_config.get("dropout", 0.1)),
    ).to(device)


def build_topk_labelwise_trust_region_model(
    *,
    previous_checkpoint_path: str | Path,
    old_bank: PrototypeBank,
    training_config: dict[str, Any],
    device: torch.device,
) -> TopKLabelWiseTrustRegionCorrectionModel:
    previous_model = load_linear_head_checkpoint(previous_checkpoint_path, device=device)
    memory_module = old_bank.to_module(
        top_k=int(training_config.get("top_k", 8)),
        temperature=float(training_config.get("temperature", 0.1)),
    ).to(device)
    return TopKLabelWiseTrustRegionCorrectionModel(
        previous_model,
        memory_module,
        feature_dim=previous_model.feature_dim,
        num_labels=previous_model.num_labels,
        top_k=int(training_config.get("correction_top_k", 1)),
        bottleneck_dim=int(training_config.get("bottleneck_dim", 128)),
        dropout=float(training_config.get("dropout", 0.1)),
    ).to(device)


def load_continual_model_checkpoint(
    *,
    checkpoint_path: str | Path,
    previous_checkpoint_path: str | Path,
    old_bank: PrototypeBank,
    training_config: dict[str, Any],
    device: torch.device,
) -> ContinualAdaptationModel:
    model = build_continual_model(
        previous_checkpoint_path=previous_checkpoint_path,
        old_bank=old_bank,
        training_config=training_config,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    return model


def _evaluate_and_write(
    model: torch.nn.Module,
    eval_datasets: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for alias, dataset in eval_datasets.items():
        metrics = evaluate_model(model, dataset, batch_size=batch_size, device=device)
        prediction_path = save_prediction_artifact(output_dir / "metrics" / f"{alias}_predictions.npz", metrics["predictions"])
        metrics_without_predictions = dict(metrics)
        metrics_without_predictions["prediction_artifact"] = str(prediction_path)
        del metrics_without_predictions["predictions"]
        write_stage_metrics(output_dir / "metrics", f"{alias}_metrics.json", metrics_without_predictions)
        payload[alias] = metrics_without_predictions
    return payload


def train_linear_adaptation(
    *,
    method_name: str,
    previous_checkpoint_path: str | Path,
    train_dataset,
    val_dataset,
    eval_datasets: dict[str, Any],
    output_dir: str | Path,
    training_config: dict[str, Any],
    device: torch.device,
    seed: int,
    fisher_state: dict[str, torch.Tensor] | None = None,
    replay_bank: PrototypeBank | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    previous_model = load_linear_head_checkpoint(previous_checkpoint_path, device=device)
    model = deepcopy(previous_model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("lr", 5.0e-4)),
        weight_decay=float(training_config.get("weight_decay", 1.0e-4)),
    )
    anchor_state = snapshot_parameters(previous_model)
    previous_model.eval()
    train_loader = build_dataloader(train_dataset, int(training_config.get("batch_size", 256)), shuffle=True, seed=seed)
    replay_loader = None
    if replay_bank is not None and replay_bank.num_prototypes > 0:
        replay_embeddings = torch.as_tensor(replay_bank.prototype_vectors, dtype=torch.float32)
        replay_targets = torch.as_tensor(replay_bank.soft_labels, dtype=torch.float32)
        replay_mask = torch.ones_like(replay_targets)
        replay_dataset = torch.utils.data.TensorDataset(replay_embeddings, replay_targets, replay_mask)
        replay_loader = torch.utils.data.DataLoader(replay_dataset, batch_size=min(64, len(replay_dataset)), shuffle=True)
    replay_iter = iter(replay_loader) if replay_loader is not None else None
    early_stopping = EarlyStopping(patience=int(training_config.get("patience", 5)), maximize=True)
    best_score = None
    history = []
    best_path = checkpoint_dir / f"{method_name}_best.pt"

    for epoch in range(int(training_config.get("epochs", 20))):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            moved = move_batch_to_device(batch, device)
            logits = model(moved["embedding"])
            loss = masked_bce_with_logits(logits, moved["target"], mask=moved["mask"])
            if method_name in {"lwf", "lwf_prototype_replay"}:
                with torch.no_grad():
                    teacher_logits = previous_model(moved["embedding"])
                loss = loss + float(training_config.get("distill_weight", 1.0)) * sigmoid_distillation_loss(
                    logits,
                    teacher_logits,
                    temperature=float(training_config.get("temperature", 2.0)),
                )
            if method_name == "l2_anchor":
                loss = loss + float(training_config.get("l2_anchor_weight", 1.0e-4)) * l2_anchor_penalty(model, anchor_state)
            if method_name == "ewc" and fisher_state is not None:
                loss = loss + float(training_config.get("ewc_weight", 1.0)) * ewc_penalty(model, anchor_state, fisher_state)
            if method_name in {"vq_summary_replay", "lwf_prototype_replay"} and replay_iter is not None:
                try:
                    replay_batch = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    replay_batch = next(replay_iter)
                replay_embeddings, replay_targets, replay_mask = [tensor.to(device) for tensor in replay_batch]
                replay_logits = model(replay_embeddings)
                replay_loss = masked_bce_with_logits(replay_logits, replay_targets, mask=replay_mask)
                loss = loss + float(training_config.get("replay_weight", 1.0)) * replay_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch["sample_id"])
            total_examples += len(batch["sample_id"])
        val_metrics = evaluate_model(model, val_dataset, batch_size=int(training_config.get("batch_size", 256)), device=device)
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
                    "feature_dim": previous_model.feature_dim,
                    "num_labels": previous_model.num_labels,
                    "best_score": best_score,
                    "history": history,
                },
                best_path,
            )
        if early_stopping.step(score):
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics_payload = _evaluate_and_write(
        model,
        eval_datasets,
        batch_size=int(training_config.get("batch_size", 256)),
        device=device,
        output_dir=output_dir,
    )
    param_counts = count_parameters(model)
    flops = {
        "linear_forward_flops": linear_layer_flops(previous_model.feature_dim, previous_model.num_labels, bias=True),
    }
    write_json(
        output_dir / "artifacts" / f"{method_name}_summary.json",
        {
            "checkpoint_path": str(best_path),
            "best_val_macro_auroc": best_score,
            "parameter_counts": param_counts,
            "flops": flops,
        },
    )
    return {
        "model": model,
        "checkpoint_path": best_path,
        "history": history,
        "best_val_macro_auroc": best_score,
        "metrics": metrics_payload,
        "parameter_counts": param_counts,
        "flops": flops,
    }


def select_fixed_alpha(
    *,
    base_model: torch.nn.Module,
    memory_module: PrototypeMemoryModule,
    val_dataset,
    batch_size: int,
    device: torch.device,
    alpha_grid: list[float],
) -> tuple[float, dict[str, Any]]:
    predictions = evaluate_model(base_model, val_dataset, batch_size=batch_size, device=device)
    base_probs = predictions["predictions"]["probabilities"]
    targets = predictions["predictions"]["targets"]
    embeddings = val_dataset.materialize_numpy()[0]
    with torch.no_grad():
        prior_probs, _ = memory_module(torch.as_tensor(embeddings, dtype=torch.float32, device=device))
    prior_probs_np = prior_probs.detach().cpu().numpy()
    best_alpha = alpha_grid[0]
    best_metrics = None
    best_score = None
    for alpha in alpha_grid:
        mixed = ((1.0 - alpha) * base_probs) + (alpha * prior_probs_np)
        metrics = evaluate_probabilities(targets, mixed)
        score = metrics.get("macro_auroc")
        if score is not None and (best_score is None or score > best_score):
            best_alpha = alpha
            best_metrics = metrics
            best_score = score
    return best_alpha, best_metrics or {}


def evaluate_probabilities(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    from paper_v1.evaluation.metrics import compute_multilabel_metrics

    mask = np.ones_like(targets, dtype=bool)
    return compute_multilabel_metrics(targets, probabilities, mask=mask)


def evaluate_fixed_alpha_mix(
    *,
    base_model: torch.nn.Module,
    memory_module: PrototypeMemoryModule,
    eval_datasets: dict[str, Any],
    alpha: float,
    batch_size: int,
    device: torch.device,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    payload: dict[str, Any] = {}
    for alias, dataset in eval_datasets.items():
        predictions = evaluate_model(base_model, dataset, batch_size=batch_size, device=device)
        base_probs = predictions["predictions"]["probabilities"]
        targets = predictions["predictions"]["targets"]
        embeddings = dataset.materialize_numpy()[0]
        with torch.no_grad():
            prior_probs, _ = memory_module(torch.as_tensor(embeddings, dtype=torch.float32, device=device))
        mixed_probs = ((1.0 - alpha) * base_probs) + (alpha * prior_probs.detach().cpu().numpy())
        metrics = evaluate_probabilities(targets, mixed_probs)
        prediction_path = output_dir / "metrics" / f"{alias}_mixed_probabilities.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            probabilities=mixed_probs,
            targets=targets,
            sample_ids=np.asarray(predictions["predictions"]["sample_ids"]),
        )
        metrics["prediction_artifact"] = str(prediction_path)
        write_stage_metrics(output_dir / "metrics", f"{alias}_metrics.json", metrics)
        payload[alias] = metrics
    return payload


def train_main_method(
    *,
    previous_checkpoint_path: str | Path,
    train_dataset,
    val_dataset,
    eval_datasets: dict[str, Any],
    old_bank: PrototypeBank,
    output_dir: str | Path,
    training_config: dict[str, Any],
    device: torch.device,
    seed: int,
    previous_model_override: torch.nn.Module | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if previous_model_override is None:
        previous_model = load_linear_head_checkpoint(previous_checkpoint_path, device=device)
        model = build_continual_model(
            previous_checkpoint_path=previous_checkpoint_path,
            old_bank=old_bank,
            training_config=training_config,
            device=device,
        )
    else:
        previous_model = previous_model_override.to(device)
        model = build_continual_model_from_previous_model(
            previous_model=previous_model,
            old_bank=old_bank,
            training_config=training_config,
            device=device,
        )
    previous_feature_dim, previous_num_labels = _infer_model_dims(previous_model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_config.get("lr", 5.0e-4)),
        weight_decay=float(training_config.get("weight_decay", 1.0e-4)),
    )
    prototype_embeddings = torch.as_tensor(old_bank.prototype_vectors, dtype=torch.float32, device=device)
    prototype_targets = torch.as_tensor(old_bank.soft_labels, dtype=torch.float32, device=device)
    prototype_mask = torch.ones_like(prototype_targets)
    train_loader = build_dataloader(train_dataset, int(training_config.get("batch_size", 256)), shuffle=True, seed=seed)
    early_stopping = EarlyStopping(patience=int(training_config.get("patience", 5)), maximize=True)
    best_score = None
    history = []
    best_path = checkpoint_dir / "main_method_best.pt"
    for epoch in range(int(training_config.get("epochs", 20))):
        model.train()
        total_loss = 0.0
        total_examples = 0
        total_current_bce = 0.0
        total_distill = 0.0
        total_replay = 0.0
        total_zero_penalty = 0.0
        gate_sum = 0.0
        gate_sq_sum = 0.0
        gate_count = 0
        raw_gate_sum = 0.0
        old_like_fraction_sum = 0.0
        residual_norm_sum = 0.0
        for batch in train_loader:
            moved = move_batch_to_device(batch, device)
            outputs = model(moved["embedding"])
            current_bce = masked_bce_with_logits(outputs["logits"], moved["target"], mask=moved["mask"])
            distill = sigmoid_distillation_loss(
                outputs["logits"],
                outputs["previous_logits"],
                temperature=float(training_config.get("distill_temperature", 2.0)),
            )
            loss = current_bce + float(training_config.get("distill_weight", 1.0)) * distill
            replay_loss = torch.zeros((), device=device)
            zero_penalty = torch.zeros((), device=device)
            if len(prototype_embeddings) > 0:
                replay_outputs = model(prototype_embeddings)
                replay_loss = masked_bce_with_logits(replay_outputs["logits"], prototype_targets, mask=prototype_mask)
                zero_penalty = torch.mean(replay_outputs["residual"] ** 2)
                loss = loss + float(training_config.get("prototype_replay_weight", 1.0)) * replay_loss
                loss = loss + float(training_config.get("residual_zero_weight", 0.1)) * zero_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch["sample_id"])
            total_examples += len(batch["sample_id"])
            total_current_bce += float(current_bce.detach().cpu()) * len(batch["sample_id"])
            total_distill += float(distill.detach().cpu()) * len(batch["sample_id"])
            total_replay += float(replay_loss.detach().cpu()) * len(batch["sample_id"])
            total_zero_penalty += float(zero_penalty.detach().cpu()) * len(batch["sample_id"])
            gate_values = outputs["gate"].detach().cpu()
            raw_gate_values = outputs["raw_gate"].detach().cpu()
            residual_norms = torch.linalg.norm(outputs["residual"].detach(), dim=1).cpu()
            gate_sum += float(gate_values.sum())
            gate_sq_sum += float(torch.sum(gate_values ** 2))
            gate_count += gate_values.numel()
            raw_gate_sum += float(raw_gate_values.sum())
            old_like_fraction_sum += float(outputs["old_like_mask"].detach().cpu().mean()) * len(batch["sample_id"])
            residual_norm_sum += float(residual_norms.sum())
        val_metrics = evaluate_model(model, val_dataset, batch_size=int(training_config.get("batch_size", 256)), device=device)
        score = val_metrics.get("macro_auroc")
        gate_mean = gate_sum / max(gate_count, 1)
        gate_var = max((gate_sq_sum / max(gate_count, 1)) - (gate_mean**2), 0.0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_examples, 1),
                "current_bce_loss": total_current_bce / max(total_examples, 1),
                "distill_loss": total_distill / max(total_examples, 1),
                "replay_loss": total_replay / max(total_examples, 1),
                "zero_residual_penalty": total_zero_penalty / max(total_examples, 1),
                "gate_mean": gate_mean,
                "gate_std": gate_var**0.5,
                "raw_gate_mean": raw_gate_sum / max(gate_count, 1),
                "old_like_fraction": old_like_fraction_sum / max(total_examples, 1),
                "residual_norm_mean": residual_norm_sum / max(total_examples, 1),
                "val_macro_auroc": score,
            }
        )
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_dim": previous_feature_dim,
                    "num_labels": previous_num_labels,
                    "best_score": best_score,
                    "history": history,
                },
                best_path,
            )
        if early_stopping.step(score):
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics_payload = _evaluate_and_write(
        model,
        eval_datasets,
        batch_size=int(training_config.get("batch_size", 256)),
        device=device,
        output_dir=output_dir,
    )
    param_counts = count_parameters(model)
    flops = {
        "residual_adapter_flops": mlp_flops(
            [
                previous_feature_dim + (2 * previous_num_labels),
                int(training_config.get("bottleneck_dim", 128)),
                previous_num_labels,
            ]
        ),
        "gate_flops": mlp_flops(
            [
                previous_feature_dim + (2 * previous_num_labels) + 2,
                int(training_config.get("gate_hidden_dim", 64)),
                1,
            ]
        ),
        "retrieval": retrieval_flops(
            previous_feature_dim,
            old_bank.num_prototypes,
            int(training_config.get("top_k", 8)),
        ),
    }
    history_csv = write_csv(output_dir / "artifacts" / "main_method_history.csv", list(history[0].keys()) if history else ["epoch"], history)
    write_json(
        output_dir / "artifacts" / "main_method_summary.json",
        {
            "checkpoint_path": str(best_path),
            "best_val_macro_auroc": best_score,
            "parameter_counts": param_counts,
            "flops": flops,
            "memory_size_bytes": old_bank.memory_size_bytes(),
            "memory_size_mb": old_bank.memory_size_bytes() / (1024.0 * 1024.0),
            "history_path": str(history_csv),
        },
    )
    return {
        "model": model,
        "checkpoint_path": best_path,
        "history": history,
        "best_val_macro_auroc": best_score,
        "metrics": metrics_payload,
        "parameter_counts": param_counts,
        "flops": flops,
    }


def train_labelwise_trust_region(
    *,
    previous_checkpoint_path: str | Path,
    train_dataset,
    val_dataset,
    eval_datasets: dict[str, Any],
    old_bank: PrototypeBank,
    output_dir: str | Path,
    training_config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    previous_model = load_linear_head_checkpoint(previous_checkpoint_path, device=device)
    model = build_labelwise_trust_region_model(
        previous_checkpoint_path=previous_checkpoint_path,
        old_bank=old_bank,
        training_config=training_config,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_config.get("lr", 5.0e-4)),
        weight_decay=float(training_config.get("weight_decay", 1.0e-4)),
    )
    prototype_embeddings = torch.as_tensor(old_bank.prototype_vectors, dtype=torch.float32, device=device)
    prototype_targets = torch.as_tensor(old_bank.soft_labels, dtype=torch.float32, device=device)
    prototype_mask = torch.ones_like(prototype_targets)
    valid_prototype_indices = [
        index
        for index, (label_index, state_name) in enumerate(zip(old_bank.label_indices, old_bank.states))
        if label_index >= 0 and state_name in {"positive", "negative"}
    ]
    if valid_prototype_indices:
        margin_label_indices = torch.as_tensor(
            [old_bank.label_indices[index] for index in valid_prototype_indices],
            dtype=torch.long,
            device=device,
        )
        margin_state_signs = torch.as_tensor(
            [1.0 if old_bank.states[index] == "positive" else -1.0 for index in valid_prototype_indices],
            dtype=torch.float32,
            device=device,
        )
        margin_embeddings = prototype_embeddings[valid_prototype_indices]
    else:
        margin_label_indices = torch.zeros((0,), dtype=torch.long, device=device)
        margin_state_signs = torch.zeros((0,), dtype=torch.float32, device=device)
        margin_embeddings = prototype_embeddings[:0]
    train_loader = build_dataloader(train_dataset, int(training_config.get("batch_size", 256)), shuffle=True, seed=seed)
    early_stopping = EarlyStopping(patience=int(training_config.get("patience", 5)), maximize=True)
    best_score = None
    history = []
    best_path = checkpoint_dir / "labelwise_trust_region_best.pt"
    gate_active_threshold = float(training_config.get("gate_active_threshold", 0.1))
    for epoch in range(int(training_config.get("epochs", 20))):
        model.train()
        total_loss = 0.0
        total_examples = 0
        total_current_bce = 0.0
        total_distill = 0.0
        total_replay = 0.0
        total_gate_sparsity = 0.0
        total_margin_preservation = 0.0
        total_trust_region = 0.0
        gate_sum = 0.0
        gate_sq_sum = 0.0
        gate_count = 0
        active_gate_sum = 0.0
        residual_norm_sum = 0.0
        trust_weight_sum = 0.0
        raw_gate_sum = 0.0
        for batch in train_loader:
            moved = move_batch_to_device(batch, device)
            outputs = model(moved["embedding"])
            current_bce = masked_bce_with_logits(outputs["logits"], moved["target"], mask=moved["mask"])
            distill = sigmoid_distillation_loss(
                outputs["logits"],
                outputs["previous_logits"],
                temperature=float(training_config.get("distill_temperature", 2.0)),
            )
            gate_penalty = gate_sparsity_loss(outputs["gate"])
            trust_region_penalty, trust_weights = residual_trust_region_loss(
                outputs["residual"],
                outputs["matched_support"],
                outputs["uncertainty"],
                support_threshold=float(training_config.get("trust_region_support_threshold", 0.7)),
            )
            loss = current_bce
            loss = loss + float(training_config.get("distill_weight", 1.0)) * distill
            loss = loss + float(training_config.get("gate_sparsity_weight", 0.05)) * gate_penalty
            loss = loss + float(training_config.get("trust_region_weight", 0.5)) * trust_region_penalty

            replay_loss = torch.zeros((), device=device)
            margin_preservation = torch.zeros((), device=device)
            if len(prototype_embeddings) > 0:
                replay_outputs = model(prototype_embeddings)
                replay_loss = masked_bce_with_logits(replay_outputs["logits"], prototype_targets, mask=prototype_mask)
                loss = loss + float(training_config.get("prototype_replay_weight", 1.0)) * replay_loss
                if len(margin_embeddings) > 0:
                    margin_outputs = replay_outputs
                    if len(valid_prototype_indices) != len(prototype_embeddings):
                        margin_outputs = model(margin_embeddings)
                    margin_preservation = old_prototype_margin_preservation_loss(
                        margin_outputs["logits"],
                        margin_outputs["previous_logits"],
                        margin_label_indices,
                        margin_state_signs,
                        slack=float(training_config.get("margin_preservation_slack", 0.25)),
                    )
                    loss = loss + float(training_config.get("margin_preservation_weight", 1.0)) * margin_preservation

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = len(batch["sample_id"])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            total_current_bce += float(current_bce.detach().cpu()) * batch_size
            total_distill += float(distill.detach().cpu()) * batch_size
            total_replay += float(replay_loss.detach().cpu()) * batch_size
            total_gate_sparsity += float(gate_penalty.detach().cpu()) * batch_size
            total_margin_preservation += float(margin_preservation.detach().cpu()) * batch_size
            total_trust_region += float(trust_region_penalty.detach().cpu()) * batch_size
            gate_values = outputs["gate"].detach().cpu()
            raw_gate_values = outputs["raw_gate"].detach().cpu()
            residual_norms = torch.linalg.norm(outputs["residual"].detach(), dim=1).cpu()
            gate_sum += float(gate_values.sum())
            gate_sq_sum += float(torch.sum(gate_values**2))
            gate_count += gate_values.numel()
            raw_gate_sum += float(raw_gate_values.sum())
            active_gate_sum += float((gate_values > gate_active_threshold).float().mean()) * batch_size
            residual_norm_sum += float(residual_norms.sum())
            trust_weight_sum += float(trust_weights.detach().cpu().mean()) * batch_size
        val_metrics = evaluate_model(model, val_dataset, batch_size=int(training_config.get("batch_size", 256)), device=device)
        score = val_metrics.get("macro_auroc")
        gate_mean = gate_sum / max(gate_count, 1)
        gate_var = max((gate_sq_sum / max(gate_count, 1)) - (gate_mean**2), 0.0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_examples, 1),
                "current_bce_loss": total_current_bce / max(total_examples, 1),
                "distill_loss": total_distill / max(total_examples, 1),
                "replay_loss": total_replay / max(total_examples, 1),
                "gate_sparsity_penalty": total_gate_sparsity / max(total_examples, 1),
                "margin_preservation_loss": total_margin_preservation / max(total_examples, 1),
                "trust_region_penalty": total_trust_region / max(total_examples, 1),
                "gate_mean": gate_mean,
                "gate_std": gate_var**0.5,
                "raw_gate_mean": raw_gate_sum / max(gate_count, 1),
                "active_label_fraction": active_gate_sum / max(total_examples, 1),
                "residual_norm_mean": residual_norm_sum / max(total_examples, 1),
                "trust_weight_mean": trust_weight_sum / max(total_examples, 1),
                "val_macro_auroc": score,
            }
        )
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_dim": previous_model.feature_dim,
                    "num_labels": previous_model.num_labels,
                    "best_score": best_score,
                    "history": history,
                },
                best_path,
            )
        if early_stopping.step(score):
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics_payload = _evaluate_and_write(
        model,
        eval_datasets,
        batch_size=int(training_config.get("batch_size", 256)),
        device=device,
        output_dir=output_dir,
    )
    param_counts = count_parameters(model)
    flops = {
        "residual_adapter_flops": mlp_flops(
            [
                previous_model.feature_dim + (2 * previous_model.num_labels),
                int(training_config.get("bottleneck_dim", 128)),
                previous_model.num_labels,
            ]
        ),
        "labelwise_gate_flops": previous_model.num_labels
        * mlp_flops(
            [
                4,
                int(training_config.get("gate_hidden_dim", 16)),
                1,
            ]
        ),
        "retrieval": retrieval_flops(
            previous_model.feature_dim,
            old_bank.num_prototypes,
            int(training_config.get("top_k", 8)),
        ),
    }
    history_csv = write_csv(
        output_dir / "artifacts" / "labelwise_trust_region_history.csv",
        list(history[0].keys()) if history else ["epoch"],
        history,
    )
    write_json(
        output_dir / "artifacts" / "labelwise_trust_region_summary.json",
        {
            "checkpoint_path": str(best_path),
            "best_val_macro_auroc": best_score,
            "parameter_counts": param_counts,
            "flops": flops,
            "memory_size_bytes": old_bank.memory_size_bytes(),
            "memory_size_mb": old_bank.memory_size_bytes() / (1024.0 * 1024.0),
            "history_path": str(history_csv),
        },
    )
    return {
        "model": model,
        "checkpoint_path": best_path,
        "history": history,
        "best_val_macro_auroc": best_score,
        "metrics": metrics_payload,
        "parameter_counts": param_counts,
        "flops": flops,
    }


def train_topk_labelwise_trust_region(
    *,
    previous_checkpoint_path: str | Path,
    train_dataset,
    val_dataset,
    eval_datasets: dict[str, Any],
    old_bank: PrototypeBank,
    output_dir: str | Path,
    training_config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    previous_model = load_linear_head_checkpoint(previous_checkpoint_path, device=device)
    model = build_topk_labelwise_trust_region_model(
        previous_checkpoint_path=previous_checkpoint_path,
        old_bank=old_bank,
        training_config=training_config,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_config.get("lr", 5.0e-4)),
        weight_decay=float(training_config.get("weight_decay", 1.0e-4)),
    )
    prototype_embeddings = torch.as_tensor(old_bank.prototype_vectors, dtype=torch.float32, device=device)
    prototype_targets = torch.as_tensor(old_bank.soft_labels, dtype=torch.float32, device=device)
    prototype_mask = torch.ones_like(prototype_targets)
    valid_prototype_indices = [
        index
        for index, (label_index, state_name) in enumerate(zip(old_bank.label_indices, old_bank.states))
        if label_index >= 0 and state_name in {"positive", "negative"}
    ]
    if valid_prototype_indices:
        margin_label_indices = torch.as_tensor(
            [old_bank.label_indices[index] for index in valid_prototype_indices],
            dtype=torch.long,
            device=device,
        )
        margin_state_signs = torch.as_tensor(
            [1.0 if old_bank.states[index] == "positive" else -1.0 for index in valid_prototype_indices],
            dtype=torch.float32,
            device=device,
        )
        margin_embeddings = prototype_embeddings[valid_prototype_indices]
    else:
        margin_label_indices = torch.zeros((0,), dtype=torch.long, device=device)
        margin_state_signs = torch.zeros((0,), dtype=torch.float32, device=device)
        margin_embeddings = prototype_embeddings[:0]
    train_loader = build_dataloader(train_dataset, int(training_config.get("batch_size", 256)), shuffle=True, seed=seed)
    early_stopping = EarlyStopping(patience=int(training_config.get("patience", 5)), maximize=True)
    best_score = None
    history = []
    method_name = f"topk_labelwise_trust_region_k{int(training_config.get('correction_top_k', 1))}"
    best_path = checkpoint_dir / f"{method_name}_best.pt"
    active_threshold = float(training_config.get("gate_active_threshold", 0.5))
    for epoch in range(int(training_config.get("epochs", 20))):
        model.train()
        total_loss = 0.0
        total_examples = 0
        total_current_bce = 0.0
        total_distill = 0.0
        total_replay = 0.0
        total_margin_preservation = 0.0
        total_trust_region = 0.0
        selection_sum = 0.0
        selection_sq_sum = 0.0
        selection_count = 0
        active_fraction_sum = 0.0
        selected_label_count_sum = 0.0
        residual_norm_sum = 0.0
        trust_weight_sum = 0.0
        score_sum = 0.0
        for batch in train_loader:
            moved = move_batch_to_device(batch, device)
            outputs = model(moved["embedding"])
            current_bce = masked_bce_with_logits(outputs["logits"], moved["target"], mask=moved["mask"])
            distill = sigmoid_distillation_loss(
                outputs["logits"],
                outputs["previous_logits"],
                temperature=float(training_config.get("distill_temperature", 2.0)),
            )
            trust_region_penalty, trust_weights = residual_trust_region_loss(
                outputs["gate"] * outputs["residual"],
                outputs["matched_support"],
                outputs["uncertainty"],
                support_threshold=float(training_config.get("trust_region_support_threshold", 0.7)),
            )
            loss = current_bce
            if float(training_config.get("distill_weight", 0.0)) != 0.0:
                loss = loss + float(training_config.get("distill_weight", 0.0)) * distill

            replay_loss = torch.zeros((), device=device)
            margin_preservation = torch.zeros((), device=device)
            if len(prototype_embeddings) > 0:
                replay_outputs = model(prototype_embeddings)
                replay_loss = masked_bce_with_logits(replay_outputs["logits"], prototype_targets, mask=prototype_mask)
                loss = loss + float(training_config.get("prototype_replay_weight", 1.0)) * replay_loss
                if len(margin_embeddings) > 0:
                    margin_outputs = replay_outputs
                    if len(valid_prototype_indices) != len(prototype_embeddings):
                        margin_outputs = model(margin_embeddings)
                    margin_preservation = old_prototype_margin_preservation_loss(
                        margin_outputs["logits"],
                        margin_outputs["previous_logits"],
                        margin_label_indices,
                        margin_state_signs,
                        slack=float(training_config.get("margin_preservation_slack", 0.25)),
                    )
                    loss = loss + float(training_config.get("margin_preservation_weight", 1.0)) * margin_preservation
            loss = loss + float(training_config.get("trust_region_weight", 0.5)) * trust_region_penalty

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = len(batch["sample_id"])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            total_current_bce += float(current_bce.detach().cpu()) * batch_size
            total_distill += float(distill.detach().cpu()) * batch_size
            total_replay += float(replay_loss.detach().cpu()) * batch_size
            total_margin_preservation += float(margin_preservation.detach().cpu()) * batch_size
            total_trust_region += float(trust_region_penalty.detach().cpu()) * batch_size
            selection_mask = outputs["gate"].detach().cpu()
            scores = outputs["raw_gate"].detach().cpu()
            selected_counts = outputs["selected_label_count"].detach().cpu()
            residual_norms = torch.linalg.norm((outputs["gate"] * outputs["residual"]).detach(), dim=1).cpu()
            selection_sum += float(selection_mask.sum())
            selection_sq_sum += float(torch.sum(selection_mask**2))
            selection_count += selection_mask.numel()
            active_fraction_sum += float((selection_mask > active_threshold).float().mean()) * batch_size
            selected_label_count_sum += float(selected_counts.sum())
            residual_norm_sum += float(residual_norms.sum())
            trust_weight_sum += float(trust_weights.detach().cpu().mean()) * batch_size
            score_sum += float(scores.sum())
        val_metrics = evaluate_model(model, val_dataset, batch_size=int(training_config.get("batch_size", 256)), device=device)
        score = val_metrics.get("macro_auroc")
        selection_mean = selection_sum / max(selection_count, 1)
        selection_var = max((selection_sq_sum / max(selection_count, 1)) - (selection_mean**2), 0.0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_examples, 1),
                "current_bce_loss": total_current_bce / max(total_examples, 1),
                "distill_loss": total_distill / max(total_examples, 1),
                "replay_loss": total_replay / max(total_examples, 1),
                "margin_preservation_loss": total_margin_preservation / max(total_examples, 1),
                "trust_region_penalty": total_trust_region / max(total_examples, 1),
                "selection_mean": selection_mean,
                "selection_std": selection_var**0.5,
                "selection_score_mean": score_sum / max(selection_count, 1),
                "active_label_fraction": active_fraction_sum / max(total_examples, 1),
                "selected_label_count_mean": selected_label_count_sum / max(total_examples, 1),
                "residual_norm_mean": residual_norm_sum / max(total_examples, 1),
                "trust_weight_mean": trust_weight_sum / max(total_examples, 1),
                "val_macro_auroc": score,
            }
        )
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_dim": previous_model.feature_dim,
                    "num_labels": previous_model.num_labels,
                    "best_score": best_score,
                    "history": history,
                },
                best_path,
            )
        if early_stopping.step(score):
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics_payload = _evaluate_and_write(
        model,
        eval_datasets,
        batch_size=int(training_config.get("batch_size", 256)),
        device=device,
        output_dir=output_dir,
    )
    param_counts = count_parameters(model)
    flops = {
        "residual_adapter_flops": mlp_flops(
            [
                previous_model.feature_dim + (2 * previous_model.num_labels),
                int(training_config.get("bottleneck_dim", 128)),
                previous_model.num_labels,
            ]
        ),
        "topk_score_flops": previous_model.num_labels * 4,
        "retrieval": retrieval_flops(
            previous_model.feature_dim,
            old_bank.num_prototypes,
            int(training_config.get("top_k", 8)),
        ),
    }
    history_csv = write_csv(
        output_dir / "artifacts" / f"{method_name}_history.csv",
        list(history[0].keys()) if history else ["epoch"],
        history,
    )
    write_json(
        output_dir / "artifacts" / f"{method_name}_summary.json",
        {
            "checkpoint_path": str(best_path),
            "best_val_macro_auroc": best_score,
            "parameter_counts": param_counts,
            "flops": flops,
            "memory_size_bytes": old_bank.memory_size_bytes(),
            "memory_size_mb": old_bank.memory_size_bytes() / (1024.0 * 1024.0),
            "history_path": str(history_csv),
            "correction_top_k": int(training_config.get("correction_top_k", 1)),
        },
    )
    return {
        "model": model,
        "checkpoint_path": best_path,
        "history": history,
        "best_val_macro_auroc": best_score,
        "metrics": metrics_payload,
        "parameter_counts": param_counts,
        "flops": flops,
    }


def train_tiny_logit_correction(
    *,
    previous_checkpoint_path: str | Path,
    train_dataset,
    val_dataset,
    eval_datasets: dict[str, Any],
    output_dir: str | Path,
    training_config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    previous_model = load_linear_head_checkpoint(previous_checkpoint_path, device=device)
    model = TinyLogitCorrection(previous_model, num_labels=previous_model.num_labels).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_config.get("lr", 1.0e-3)),
        weight_decay=float(training_config.get("weight_decay", 1.0e-5)),
    )
    train_loader = build_dataloader(train_dataset, int(training_config.get("batch_size", 256)), shuffle=True, seed=seed)
    early_stopping = EarlyStopping(patience=int(training_config.get("patience", 5)), maximize=True)
    history = []
    best_score = None
    best_path = checkpoint_dir / "tiny_logit_correction_best.pt"
    for epoch in range(int(training_config.get("epochs", 20))):
        model.train()
        total_loss = 0.0
        total_examples = 0
        total_current_bce = 0.0
        total_distill = 0.0
        total_identity = 0.0
        for batch in train_loader:
            moved = move_batch_to_device(batch, device)
            outputs = model(moved["embedding"])
            current_bce = masked_bce_with_logits(outputs["logits"], moved["target"], mask=moved["mask"])
            distill = sigmoid_distillation_loss(
                outputs["logits"],
                outputs["previous_logits"],
                temperature=float(training_config.get("distill_temperature", 2.0)),
            )
            identity_penalty = torch.mean((model.logit_scale - 1.0) ** 2) + torch.mean(model.logit_bias**2)
            loss = current_bce
            loss = loss + float(training_config.get("distill_weight", 1.0)) * distill
            loss = loss + float(training_config.get("identity_weight", 0.1)) * identity_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch["sample_id"])
            total_examples += len(batch["sample_id"])
            total_current_bce += float(current_bce.detach().cpu()) * len(batch["sample_id"])
            total_distill += float(distill.detach().cpu()) * len(batch["sample_id"])
            total_identity += float(identity_penalty.detach().cpu()) * len(batch["sample_id"])
        val_metrics = evaluate_model(model, val_dataset, batch_size=int(training_config.get("batch_size", 256)), device=device)
        score = val_metrics.get("macro_auroc")
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_examples, 1),
                "current_bce_loss": total_current_bce / max(total_examples, 1),
                "distill_loss": total_distill / max(total_examples, 1),
                "identity_penalty": total_identity / max(total_examples, 1),
                "logit_scale_mean": float(model.logit_scale.detach().cpu().mean()),
                "logit_bias_mean": float(model.logit_bias.detach().cpu().mean()),
                "val_macro_auroc": score,
            }
        )
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_dim": previous_model.feature_dim,
                    "num_labels": previous_model.num_labels,
                    "best_score": best_score,
                    "history": history,
                },
                best_path,
            )
        if early_stopping.step(score):
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics_payload = _evaluate_and_write(
        model,
        eval_datasets,
        batch_size=int(training_config.get("batch_size", 256)),
        device=device,
        output_dir=output_dir,
    )
    param_counts = count_parameters(model)
    history_csv = write_csv(
        output_dir / "artifacts" / "tiny_logit_correction_history.csv",
        list(history[0].keys()) if history else ["epoch"],
        history,
    )
    write_json(
        output_dir / "artifacts" / "tiny_logit_correction_summary.json",
        {
            "checkpoint_path": str(best_path),
            "best_val_macro_auroc": best_score,
            "parameter_counts": param_counts,
            "flops": {
                "affine_logit_correction_flops": 3 * previous_model.num_labels,
            },
            "history_path": str(history_csv),
        },
    )
    return {
        "model": model,
        "checkpoint_path": best_path,
        "history": history,
        "best_val_macro_auroc": best_score,
        "metrics": metrics_payload,
        "parameter_counts": param_counts,
        "flops": {
            "affine_logit_correction_flops": 3 * previous_model.num_labels,
        },
    }

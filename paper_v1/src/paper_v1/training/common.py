"""Shared training and evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper_v1.evaluation.metrics import compute_multilabel_metrics


def build_dataloader(dataset, batch_size: int, *, shuffle: bool, seed: int = 1337) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def predict_model(model: torch.nn.Module, dataset, *, batch_size: int, device: torch.device) -> dict[str, Any]:
    dataloader = build_dataloader(dataset, batch_size, shuffle=False)
    model.eval()
    logits_chunks = []
    probs_chunks = []
    targets_chunks = []
    masks_chunks = []
    sample_ids: list[str] = []
    for batch in dataloader:
        moved = move_batch_to_device(batch, device)
        outputs = model(moved["embedding"])
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs
        probabilities = torch.sigmoid(logits)
        logits_chunks.append(logits.detach().cpu().numpy())
        probs_chunks.append(probabilities.detach().cpu().numpy())
        targets_chunks.append(moved["target"].detach().cpu().numpy())
        masks_chunks.append(moved["mask"].detach().cpu().numpy())
        sample_ids.extend(list(batch["sample_id"]))
    logits = np.concatenate(logits_chunks, axis=0) if logits_chunks else np.zeros((0, 0), dtype=np.float32)
    probabilities = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.zeros((0, 0), dtype=np.float32)
    targets = np.concatenate(targets_chunks, axis=0) if targets_chunks else np.zeros((0, 0), dtype=np.float32)
    masks = np.concatenate(masks_chunks, axis=0) if masks_chunks else np.zeros((0, 0), dtype=np.float32)
    return {
        "logits": logits,
        "probabilities": probabilities,
        "targets": targets,
        "masks": masks,
        "sample_ids": sample_ids,
    }


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dataset, *, batch_size: int, device: torch.device) -> dict[str, Any]:
    predictions = predict_model(model, dataset, batch_size=batch_size, device=device)
    metrics = compute_multilabel_metrics(
        predictions["targets"],
        predictions["probabilities"],
        mask=predictions["masks"].astype(bool),
    )
    metrics["predictions"] = predictions
    return metrics


def save_prediction_artifact(path: str | Path, predictions: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        logits=predictions["logits"],
        probabilities=predictions["probabilities"],
        targets=predictions["targets"],
        masks=predictions["masks"],
        sample_ids=np.asarray(predictions["sample_ids"]),
    )
    return output_path

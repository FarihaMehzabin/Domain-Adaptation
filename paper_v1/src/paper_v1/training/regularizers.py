"""Regularizers and Fisher estimation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch.utils.data import DataLoader

from paper_v1.training.losses import masked_bce_with_logits


def snapshot_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def l2_anchor_penalty(model: torch.nn.Module, anchor_state: dict[str, torch.Tensor]) -> torch.Tensor:
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, parameter in model.named_parameters():
        penalty = penalty + torch.sum((parameter - anchor_state[name].to(parameter.device)) ** 2)
    return penalty


def ewc_penalty(
    model: torch.nn.Module,
    anchor_state: dict[str, torch.Tensor],
    fisher_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, parameter in model.named_parameters():
        fisher = fisher_state[name].to(parameter.device)
        anchor = anchor_state[name].to(parameter.device)
        penalty = penalty + torch.sum(fisher * ((parameter - anchor) ** 2))
    return penalty


def compute_fisher_diagonal(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, torch.Tensor]:
    fisher = {name: torch.zeros_like(parameter, device=device) for name, parameter in model.named_parameters()}
    model.eval()
    num_batches = 0
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        embeddings = batch["embedding"].to(device)
        targets = batch["target"].to(device)
        mask = batch["mask"].to(device)
        logits = model(embeddings)
        if isinstance(logits, dict):
            logits = logits["logits"]
        loss = masked_bce_with_logits(logits, targets, mask=mask)
        model.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            fisher[name] += parameter.grad.detach() ** 2
        num_batches += 1
    if num_batches == 0:
        return fisher
    for name in fisher:
        fisher[name] = fisher[name] / num_batches
    return {name: tensor.detach().cpu() for name, tensor in fisher.items()}

"""Parameter accounting."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def count_parameters(module: nn.Module) -> dict[str, Any]:
    trainable = 0
    frozen = 0
    per_parameter = []
    for name, parameter in module.named_parameters():
        count = parameter.numel()
        if parameter.requires_grad:
            trainable += count
        else:
            frozen += count
        per_parameter.append(
            {
                "name": name,
                "count": count,
                "trainable": bool(parameter.requires_grad),
            }
        )
    return {
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "total_parameters": trainable + frozen,
        "by_parameter": per_parameter,
    }

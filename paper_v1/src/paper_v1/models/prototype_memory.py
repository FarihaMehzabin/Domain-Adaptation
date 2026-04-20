"""Compact prototype memory and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from paper_v1.data.label_space import DEFAULT_LABEL_SPACE
from paper_v1.utils.io import write_json


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return array / norms


def _spherical_kmeans(vectors: np.ndarray, k: int, *, seed: int, max_iter: int = 20) -> tuple[np.ndarray, np.ndarray]:
    if len(vectors) == 0:
        return np.zeros((0, vectors.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    normalized = _normalize_rows(vectors.astype(np.float32))
    if len(normalized) <= k:
        assignments = np.arange(len(normalized), dtype=np.int64)
        return normalized.copy(), assignments
    rng = np.random.default_rng(seed)
    initial_indices = rng.choice(len(normalized), size=k, replace=False)
    centers = normalized[initial_indices]
    assignments = np.zeros(len(normalized), dtype=np.int64)
    for _ in range(max_iter):
        similarities = normalized @ centers.T
        next_assignments = similarities.argmax(axis=1)
        if np.array_equal(assignments, next_assignments):
            break
        assignments = next_assignments
        updated = []
        for cluster_index in range(k):
            cluster_vectors = normalized[assignments == cluster_index]
            if len(cluster_vectors) == 0:
                updated.append(centers[cluster_index])
                continue
            mean_vector = cluster_vectors.mean(axis=0, keepdims=True)
            updated.append(_normalize_rows(mean_vector)[0])
        centers = np.stack(updated).astype(np.float32)
    return centers, assignments


@dataclass(frozen=True)
class PrototypeBank:
    prototype_vectors: np.ndarray
    soft_labels: np.ndarray
    cluster_counts: np.ndarray
    source_domains: list[str]
    label_indices: list[int]
    states: list[str]
    label_names: list[str]

    @property
    def num_prototypes(self) -> int:
        return int(self.prototype_vectors.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.prototype_vectors.shape[1]) if self.num_prototypes else 0

    @property
    def num_labels(self) -> int:
        return int(self.soft_labels.shape[1]) if self.num_prototypes else len(self.label_names)

    def memory_size_bytes(self) -> int:
        metadata_bytes = 0
        for domain in self.source_domains:
            metadata_bytes += len(domain.encode("utf-8"))
        for state in self.states:
            metadata_bytes += len(state.encode("utf-8"))
        metadata_bytes += 8 * len(self.label_indices)
        return int(
            self.prototype_vectors.nbytes
            + self.soft_labels.nbytes
            + self.cluster_counts.nbytes
            + metadata_bytes
        )

    def to_module(self, *, top_k: int = 8, temperature: float = 0.1) -> "PrototypeMemoryModule":
        return PrototypeMemoryModule(self, top_k=top_k, temperature=temperature)

    def save(self, output_prefix: str | Path) -> tuple[Path, Path]:
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        npz_path = output_prefix.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            prototype_vectors=self.prototype_vectors,
            soft_labels=self.soft_labels,
            cluster_counts=self.cluster_counts,
        )
        meta_path = output_prefix.with_suffix(".json")
        write_json(
            meta_path,
            {
                "source_domains": self.source_domains,
                "label_indices": self.label_indices,
                "states": self.states,
                "label_names": self.label_names,
                "memory_size_bytes": self.memory_size_bytes(),
                "memory_size_mb": self.memory_size_bytes() / (1024.0 * 1024.0),
            },
        )
        return npz_path, meta_path

    @classmethod
    def load(cls, npz_path: str | Path, meta_path: str | Path) -> "PrototypeBank":
        arrays = np.load(npz_path)
        from paper_v1.utils.io import read_json

        metadata = read_json(meta_path)
        return cls(
            prototype_vectors=np.asarray(arrays["prototype_vectors"], dtype=np.float32),
            soft_labels=np.asarray(arrays["soft_labels"], dtype=np.float32),
            cluster_counts=np.asarray(arrays["cluster_counts"], dtype=np.int64),
            source_domains=list(metadata["source_domains"]),
            label_indices=list(metadata["label_indices"]),
            states=list(metadata["states"]),
            label_names=list(metadata["label_names"]),
        )


def merge_prototype_banks(banks: list[PrototypeBank]) -> PrototypeBank:
    non_empty = [bank for bank in banks if bank.num_prototypes > 0]
    if not non_empty:
        return PrototypeBank(
            prototype_vectors=np.zeros((0, 0), dtype=np.float32),
            soft_labels=np.zeros((0, len(DEFAULT_LABEL_SPACE.names)), dtype=np.float32),
            cluster_counts=np.zeros((0,), dtype=np.int64),
            source_domains=[],
            label_indices=[],
            states=[],
            label_names=list(DEFAULT_LABEL_SPACE.names),
        )
    feature_dim = non_empty[0].feature_dim
    label_names = non_empty[0].label_names
    for bank in non_empty[1:]:
        if bank.feature_dim != feature_dim:
            raise ValueError("all prototype banks must share the same feature dimension")
        if bank.label_names != label_names:
            raise ValueError("all prototype banks must share the same label names")
    return PrototypeBank(
        prototype_vectors=np.concatenate([bank.prototype_vectors for bank in non_empty], axis=0).astype(np.float32),
        soft_labels=np.concatenate([bank.soft_labels for bank in non_empty], axis=0).astype(np.float32),
        cluster_counts=np.concatenate([bank.cluster_counts for bank in non_empty], axis=0).astype(np.int64),
        source_domains=[domain for bank in non_empty for domain in bank.source_domains],
        label_indices=[label_index for bank in non_empty for label_index in bank.label_indices],
        states=[state for bank in non_empty for state in bank.states],
        label_names=list(label_names),
    )


class PrototypeMemoryModule(nn.Module):
    def __init__(self, bank: PrototypeBank, *, top_k: int = 8, temperature: float = 0.1) -> None:
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        vectors = torch.as_tensor(bank.prototype_vectors, dtype=torch.float32)
        labels = torch.as_tensor(bank.soft_labels, dtype=torch.float32)
        self.label_names = list(bank.label_names[: labels.shape[1]])
        label_indices = torch.as_tensor(bank.label_indices, dtype=torch.long)
        state_values = torch.as_tensor(
            [
                1 if state == "positive" else 0 if state == "negative" else -1
                for state in bank.states
            ],
            dtype=torch.long,
        )
        self.register_buffer("prototype_vectors", vectors)
        self.register_buffer("soft_labels", labels)
        self.register_buffer("label_indices", label_indices)
        self.register_buffer("state_values", state_values)

    def _empty_outputs(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = embeddings.shape[0]
        num_labels = len(self.label_names)
        zeros = torch.zeros(batch_size, num_labels, device=embeddings.device, dtype=embeddings.dtype)
        stats = torch.zeros(batch_size, 2, device=embeddings.device, dtype=embeddings.dtype)
        return {
            "prior_probabilities": zeros,
            "retrieval_summary": stats,
            "positive_support": torch.full_like(zeros, -1.0),
            "negative_support": torch.full_like(zeros, -1.0),
            "matched_support": torch.full_like(zeros, -1.0),
            "opposing_support": torch.full_like(zeros, -1.0),
            "matched_distance": torch.full_like(zeros, 2.0),
            "support_margin": torch.zeros_like(zeros),
        }

    def retrieve(
        self,
        embeddings: torch.Tensor,
        *,
        previous_probabilities: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.prototype_vectors.numel() == 0:
            return self._empty_outputs(embeddings)
        query = torch.nn.functional.normalize(embeddings, dim=1)
        bank = torch.nn.functional.normalize(self.prototype_vectors, dim=1)
        similarities = query @ bank.T
        top_k = min(self.top_k, similarities.shape[1])
        top_values, top_indices = torch.topk(similarities, k=top_k, dim=1)
        weights = torch.softmax(top_values / max(self.temperature, 1.0e-6), dim=1)
        selected_labels = self.soft_labels[top_indices]
        prior = torch.sum(weights.unsqueeze(-1) * selected_labels, dim=1)
        retrieval_stats = torch.stack([top_values.max(dim=1).values, top_values.mean(dim=1)], dim=1)
        positive_support = []
        negative_support = []
        for label_index in range(len(self.label_names)):
            positive_mask = (self.label_indices == label_index) & (self.state_values == 1)
            negative_mask = (self.label_indices == label_index) & (self.state_values == 0)
            if torch.any(positive_mask):
                positive_support.append(similarities[:, positive_mask].max(dim=1).values)
            else:
                positive_support.append(torch.full((embeddings.shape[0],), -1.0, device=embeddings.device, dtype=embeddings.dtype))
            if torch.any(negative_mask):
                negative_support.append(similarities[:, negative_mask].max(dim=1).values)
            else:
                negative_support.append(torch.full((embeddings.shape[0],), -1.0, device=embeddings.device, dtype=embeddings.dtype))
        positive_support_tensor = torch.stack(positive_support, dim=1)
        negative_support_tensor = torch.stack(negative_support, dim=1)
        if previous_probabilities is None:
            previous_probabilities = prior
        predicted_positive = previous_probabilities >= 0.5
        matched_support = torch.where(predicted_positive, positive_support_tensor, negative_support_tensor)
        opposing_support = torch.where(predicted_positive, negative_support_tensor, positive_support_tensor)
        matched_distance = 1.0 - matched_support
        support_margin = matched_support - opposing_support
        return {
            "prior_probabilities": prior,
            "retrieval_summary": retrieval_stats,
            "positive_support": positive_support_tensor,
            "negative_support": negative_support_tensor,
            "matched_support": matched_support,
            "opposing_support": opposing_support,
            "matched_distance": matched_distance,
            "support_margin": support_margin,
        }

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        retrieval = self.retrieve(embeddings)
        return retrieval["prior_probabilities"], retrieval["retrieval_summary"]


def build_label_state_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    domain: str,
    positive_k: int = 4,
    negative_k: int = 2,
    seed: int = 1337,
) -> PrototypeBank:
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must share the first dimension")
    vectors: list[np.ndarray] = []
    soft_labels: list[np.ndarray] = []
    cluster_counts: list[int] = []
    source_domains: list[str] = []
    label_indices: list[int] = []
    states: list[str] = []
    for label_index in range(labels.shape[1]):
        for state_name, state_value, k in (("positive", 1.0, positive_k), ("negative", 0.0, negative_k)):
            mask = labels[:, label_index] == state_value
            subset_embeddings = embeddings[mask]
            subset_labels = labels[mask]
            if len(subset_embeddings) == 0:
                continue
            centers, assignments = _spherical_kmeans(subset_embeddings, k=min(k, len(subset_embeddings)), seed=seed + label_index)
            for cluster_index in range(centers.shape[0]):
                cluster_mask = assignments == cluster_index
                cluster_vectors = subset_embeddings[cluster_mask]
                cluster_targets = subset_labels[cluster_mask]
                if len(cluster_vectors) == 0:
                    continue
                vectors.append(centers[cluster_index].astype(np.float32))
                soft_labels.append(cluster_targets.mean(axis=0).astype(np.float32))
                cluster_counts.append(int(len(cluster_vectors)))
                source_domains.append(domain)
                label_indices.append(label_index)
                states.append(state_name)
    if not vectors:
        return PrototypeBank(
            prototype_vectors=np.zeros((0, embeddings.shape[1]), dtype=np.float32),
            soft_labels=np.zeros((0, labels.shape[1]), dtype=np.float32),
            cluster_counts=np.zeros((0,), dtype=np.int64),
            source_domains=[],
            label_indices=[],
            states=[],
            label_names=list(DEFAULT_LABEL_SPACE.names),
        )
    return PrototypeBank(
        prototype_vectors=np.stack(vectors).astype(np.float32),
        soft_labels=np.stack(soft_labels).astype(np.float32),
        cluster_counts=np.asarray(cluster_counts, dtype=np.int64),
        source_domains=source_domains,
        label_indices=label_indices,
        states=states,
        label_names=list(DEFAULT_LABEL_SPACE.names),
    )


def build_vq_summary_bank(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    budget_bytes: int,
    seed: int = 1337,
) -> PrototypeBank:
    bytes_per_prototype = 4 * (embeddings.shape[1] + labels.shape[1])
    num_prototypes = max(1, budget_bytes // max(bytes_per_prototype, 1))
    centers, assignments = _spherical_kmeans(embeddings, k=min(num_prototypes, len(embeddings)), seed=seed)
    vectors: list[np.ndarray] = []
    soft_labels: list[np.ndarray] = []
    cluster_counts: list[int] = []
    for cluster_index in range(centers.shape[0]):
        cluster_mask = assignments == cluster_index
        cluster_vectors = embeddings[cluster_mask]
        cluster_targets = labels[cluster_mask]
        vectors.append(centers[cluster_index].astype(np.float32))
        soft_labels.append(cluster_targets.mean(axis=0).astype(np.float32))
        cluster_counts.append(int(cluster_mask.sum()))
    return PrototypeBank(
        prototype_vectors=np.stack(vectors).astype(np.float32),
        soft_labels=np.stack(soft_labels).astype(np.float32),
        cluster_counts=np.asarray(cluster_counts, dtype=np.int64),
        source_domains=["vq_summary"] * len(vectors),
        label_indices=[-1] * len(vectors),
        states=["summary"] * len(vectors),
        label_names=list(DEFAULT_LABEL_SPACE.names),
    )

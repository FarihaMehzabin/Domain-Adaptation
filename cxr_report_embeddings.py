#!/usr/bin/env python3
"""Beginner-friendly HF embedding pipeline for chest X-ray reports.

Implements:
- encode_reports(reports: list[str], model_id: str) -> np.ndarray[float32]
- L2-normalized embeddings
- FAISS cosine retrieval via IndexFlatIP
- per-model artifact saving (embeddings.npy, report_ids.json, model_meta.json)
- smoke tests + retrieval diagnostics for model comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    from packaging.version import Version
except Exception:  # pragma: no cover
    Version = None  # type: ignore

DEFAULT_MODELS = [
    "microsoft/BiomedVLP-CXR-BERT-specialized",
    "NeuML/bioclinical-modernbert-base-embeddings",
    "sentence-transformers/all-MiniLM-L6-v2",
]

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "microsoft/BiomedVLP-CXR-BERT-specialized": {
        "pooling": "cls",
        "max_length": 512,
    },
    "NeuML/bioclinical-modernbert-base-embeddings": {
        "pooling": "mean",
        "max_length": 1024,
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "pooling": "mean",
        "max_length": 256,
    },
}

MODEL_DEPENDENCY_PROFILES: dict[str, list[str]] = {
    # Known stable combination for CXR-BERT remote-code loading.
    "microsoft/BiomedVLP-CXR-BERT-specialized": [
        "transformers==4.45.2",
        "accelerate==0.34.2",
        "safetensors==0.4.5",
        "packaging==24.1",
        "huggingface-hub==0.25.2",
        "tokenizers==0.20.1",
    ],
    # ModernBERT requires a newer transformers runtime.
    "NeuML/bioclinical-modernbert-base-embeddings": [
        "transformers==4.57.6",
        "accelerate==1.13.0",
        "safetensors==0.7.0",
        "packaging==24.1",
        "huggingface-hub==0.36.2",
        "tokenizers==0.22.2",
    ],
    "sentence-transformers/all-MiniLM-L6-v2": [
        "transformers==4.57.6",
        "sentence-transformers==5.2.3",
        "accelerate==1.13.0",
        "safetensors==0.7.0",
        "packaging==24.1",
        "huggingface-hub==0.36.2",
        "tokenizers==0.22.2",
    ],
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class ReportRecord:
    report_id: str
    text: str
    label: Optional[str] = None


@dataclass
class RunMetrics:
    model_id: str
    output_dir: str
    embedding_dim: int
    num_reports: int
    shape_ok: bool
    nan_inf_ok: bool
    encode_seconds: float
    reports_per_second: float
    self_retrieval_top1_rate: float
    self_retrieval_top1_score_mean: float
    batch_stability_max_abs_diff: float
    batch_stable: bool
    mean_lexical_overlap_top5: float
    majority_label_top5_accuracy: Optional[float]
    majority_label_eval_count: int
    label_join_coverage: Optional[float] = None
    label_topk_any_overlap_rate: Optional[float] = None
    label_topk_mean_jaccard: Optional[float] = None
    label_majority_topk_accuracy: Optional[float] = None
    label_majority_eval_count: int = 0
    label_unmatched_sample_ids: list[str] | None = None
    nan_retry_used: bool = False
    precision_mode: Optional[str] = None
    nonfinite_rows_initial: int = 0
    nonfinite_rows_final: int = 0
    nonfinite_rows_reencoded: int = 0
    nonfinite_rows_zero_filled: int = 0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def _require_module(module_obj: Any, package_name: str, install_hint: str) -> None:
    if module_obj is None:
        raise SystemExit(
            f"Missing dependency '{package_name}'. Install with: {install_hint}"
        )


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return embeddings / norms


def _nonfinite_row_indices(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        return np.array([], dtype=np.int64)
    mask = ~np.isfinite(embeddings).all(axis=1)
    return np.where(mask)[0]


def _tail_text(text: str, max_lines: int = 80) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _installed_version(pkg: str) -> Optional[str]:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def _ensure_runtime_compat_for_model(model_id: str) -> None:
    _require_module(Version, "packaging", "pip install packaging")
    tf_ver = _installed_version("transformers")
    if tf_ver is None:
        return
    torch_ver = _installed_version("torch")

    assert Version is not None
    if Version(tf_ver) >= Version("5.0.0"):
        if torch_ver is None or Version(torch_ver) < Version("2.4.0"):
            raise RuntimeError(
                f"transformers {tf_ver} is incompatible with torch {torch_ver or 'missing'} in this pipeline. "
                "Use pinned deps (install-per-model) or: "
                "pip install -U 'transformers==4.57.6' 'accelerate==1.13.0' 'safetensors==0.7.0' "
                "'packaging==24.1' 'huggingface-hub==0.36.2' 'tokenizers==0.22.2'"
            )

    low = model_id.lower()
    if "modernbert" in low:
        if Version(tf_ver) < Version("4.48.0"):
            raise RuntimeError(
                f"Model '{model_id}' requires transformers>=4.48.0 (found {tf_ver}). "
                "Run pinned stack: "
                "pip install -U 'transformers==4.57.6' 'accelerate==1.13.0' 'safetensors==0.7.0' "
                "'packaging==24.1' 'huggingface-hub==0.36.2' 'tokenizers==0.22.2'"
            )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    if torch is None:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _maybe_clear_cuda_cache() -> None:
    if torch is None:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _dependency_profile_for_model(model_id: str) -> list[str]:
    if model_id in MODEL_DEPENDENCY_PROFILES:
        return MODEL_DEPENDENCY_PROFILES[model_id]
    low = model_id.lower()
    if "modernbert" in low:
        return [
            "transformers==4.57.6",
            "accelerate==1.13.0",
            "safetensors==0.7.0",
            "packaging==24.1",
            "huggingface-hub==0.36.2",
            "tokenizers==0.22.2",
        ]
    return [
        "transformers==4.57.6",
        "accelerate==1.13.0",
        "safetensors==0.7.0",
        "packaging==24.1",
        "huggingface-hub==0.36.2",
        "tokenizers==0.22.2",
    ]


def _install_dependencies_for_model(model_id: str, python_bin: str) -> list[str]:
    reqs = _dependency_profile_for_model(model_id)
    cmd = [python_bin, "-m", "pip", "install", "-U", *reqs]
    print(f"[deps] model={model_id}")
    print(f"[deps] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return reqs


def _infer_pooling(model_id: str, pooling: Optional[str]) -> str:
    if pooling:
        return pooling
    return str(MODEL_SPECS.get(model_id, {}).get("pooling", "mean"))


def _infer_max_length(model_id: str, max_length: Optional[int]) -> Optional[int]:
    if max_length is not None:
        return max_length
    value = MODEL_SPECS.get(model_id, {}).get("max_length")
    return int(value) if value is not None else None


def _pool_hidden(
    outputs: Any,
    attention_mask: Optional[Any],
    pooling: str,
) -> Any:
    # Some models expose a direct pooled text embedding.
    text_embeds = getattr(outputs, "text_embeds", None)
    if text_embeds is not None:
        return text_embeds

    if pooling == "pooler":
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output

    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is None and isinstance(outputs, tuple) and outputs:
        if torch is not None and torch.is_tensor(outputs[0]):
            last_hidden = outputs[0]

    if last_hidden is None:
        raise RuntimeError("Could not locate hidden states for pooling.")

    if last_hidden.ndim == 2:
        return last_hidden

    if pooling == "cls":
        return last_hidden[:, 0, :]

    if attention_mask is None:
        return last_hidden.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-9)
    return summed / counts


def _model_has_meta_tensors(model: Any) -> bool:
    if torch is None:
        return False
    for param in model.parameters():
        if getattr(param, "is_meta", False):
            return True
    for buff in model.buffers():
        if getattr(buff, "is_meta", False):
            return True
    return False


def _infer_model_input_device(model: Any, fallback: str) -> str:
    if torch is None:
        return fallback
    for param in model.parameters():
        if not getattr(param, "is_meta", False):
            return str(param.device)
    for buff in model.buffers():
        if not getattr(buff, "is_meta", False):
            return str(buff.device)
    return fallback


class HFTextEncoder:
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        pooling: Optional[str] = None,
        max_length: Optional[int] = None,
        trust_remote_code: bool = True,
        prefer_sentence_transformers: bool = False,
        force_fp32: bool = False,
    ):
        self.model_id = model_id
        self.device = _resolve_device(device)
        self.pooling = _infer_pooling(model_id, pooling)
        self.max_length = _infer_max_length(model_id, max_length)
        self.trust_remote_code = trust_remote_code
        self.prefer_sentence_transformers = prefer_sentence_transformers
        self.force_fp32 = force_fp32

        self.backend = "transformers"
        self.tokenizer = None
        self.model = None
        self.st_model = None
        self.use_projected_text_embeddings = False
        self.precision_mode = "unknown"

        if self.prefer_sentence_transformers and SentenceTransformer is not None:
            self._load_sentence_transformers_backend()
        else:
            self._load_transformers_backend()

    def _load_sentence_transformers_backend(self) -> None:
        _require_module(
            SentenceTransformer,
            "sentence-transformers",
            "pip install sentence-transformers",
        )
        self.st_model = SentenceTransformer(
            self.model_id,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
        )
        self.backend = "sentence_transformers"
        self.precision_mode = "st_default"

    def _load_transformers_backend(self) -> None:
        _require_module(torch, "torch", "pip install torch")
        _require_module(
            AutoTokenizer,
            "transformers",
            "pip install transformers",
        )
        _require_module(
            AutoModel,
            "transformers",
            "pip install transformers",
        )

        assert torch is not None
        assert AutoTokenizer is not None
        assert AutoModel is not None

        _ensure_runtime_compat_for_model(self.model_id)

        torch_dtype = None
        if self.force_fp32:
            torch_dtype = torch.float32
            self.precision_mode = "fp32"
        elif self.device == "cuda":
            if "modernbert" in self.model_id.lower():
                # ModernBERT is often numerically fragile in fp16/bf16 for embeddings.
                torch_dtype = torch.float32
                self.precision_mode = "fp32"
            elif torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
                self.precision_mode = "bf16"
            else:
                torch_dtype = torch.float16
                self.precision_mode = "fp16"
        else:
            self.precision_mode = "fp32"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

        model_kwargs = dict(
            trust_remote_code=self.trust_remote_code,
            # Avoid meta-init loading path for fragile custom remote-code models.
            low_cpu_mem_usage=False,
        )
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        attempts: list[tuple[str, dict[str, Any], bool, Optional[str]]] = [
            ("primary", dict(model_kwargs), True, self.device),
            # Fallback 1: reload in float32 in case dtype-triggered loading is brittle.
            (
                "fp32_retry",
                {"trust_remote_code": self.trust_remote_code, "low_cpu_mem_usage": False},
                True,
                self.device,
            ),
        ]

        # Fallback 2: use device_map loading path, which can resolve meta-only tensors.
        if self.device == "cuda":
            dev_map: Any = "auto"
        else:
            # Keep mps as regular .to() path; device_map on mps is less reliable.
            dev_map = {"": "cpu"}

        device_map_kwargs = dict(model_kwargs)
        device_map_kwargs["low_cpu_mem_usage"] = True
        device_map_kwargs["device_map"] = dev_map
        attempts.append(("device_map_retry", device_map_kwargs, False, None))

        last_error: Optional[Exception] = None
        loaded_model = None
        loaded_device = self.device
        for attempt_name, load_kwargs, requires_to, target_device in attempts:
            try:
                candidate = AutoModel.from_pretrained(
                    self.model_id,
                    **load_kwargs,
                )
                if requires_to:
                    move_target = target_device or self.device
                    candidate.to(move_target)
                if _model_has_meta_tensors(candidate):
                    raise RuntimeError(
                        f"Model still contains meta tensors after '{attempt_name}' load."
                    )
                loaded_model = candidate
                loaded_device = _infer_model_input_device(candidate, target_device or self.device)
                break
            except Exception as exc:  # pragma: no cover - runtime env/model specific
                last_error = exc
                print(f"[warn] load attempt '{attempt_name}' failed: {exc}")

        if loaded_model is None:
            raise RuntimeError(
                f"Failed to load model '{self.model_id}' on device='{self.device}'. "
                "Check model/runtime compatibility (e.g., ModernBERT needs newer transformers)."
            ) from last_error

        self.model = loaded_model
        self.device = loaded_device
        self.model.eval()
        self.backend = "transformers"
        self.use_projected_text_embeddings = hasattr(self.model, "get_projected_text_embeddings")
        if self.use_projected_text_embeddings:
            self.pooling = "projected_cls"

    def encode(
        self,
        reports: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        if not reports:
            return np.zeros((0, 0), dtype=np.float32)

        if self.backend == "sentence_transformers":
            assert self.st_model is not None
            embeddings = self.st_model.encode(
                reports,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            return embeddings.astype(np.float32)

        assert torch is not None
        assert self.tokenizer is not None
        assert self.model is not None

        chunks: list[np.ndarray] = []
        for start in range(0, len(reports), batch_size):
            batch = reports[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.inference_mode():
                if self.use_projected_text_embeddings:
                    projected = self.model.get_projected_text_embeddings(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        normalize_embeddings=False,
                    )
                    pooled = projected
                else:
                    outputs = self.model(**encoded)
                    pooled = _pool_hidden(
                        outputs=outputs,
                        attention_mask=encoded.get("attention_mask"),
                        pooling=self.pooling,
                    )
            chunks.append(pooled.detach().float().cpu().numpy().astype(np.float32))

        embeddings = np.concatenate(chunks, axis=0)
        if normalize:
            embeddings = _l2_normalize(embeddings)
        return embeddings.astype(np.float32)


# Required interface from the implementation plan.
def encode_reports(reports: list[str], model_id: str) -> np.ndarray:
    _require_module(np, "numpy", "pip install numpy")
    encoder = HFTextEncoder(model_id=model_id)
    return encoder.encode(reports, batch_size=32, normalize=True)


def _load_reports_from_csv(
    csv_path: Path,
    text_col: str,
    id_col: Optional[str],
    label_col: Optional[str],
    limit: Optional[int],
    min_chars: int,
) -> list[ReportRecord]:
    rows: list[ReportRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        if text_col not in reader.fieldnames:
            raise ValueError(
                f"text_col '{text_col}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        if id_col and id_col not in reader.fieldnames:
            raise ValueError(
                f"id_col '{id_col}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        if label_col and label_col not in reader.fieldnames:
            raise ValueError(
                f"label_col '{label_col}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        for idx, row in enumerate(reader):
            text = (row.get(text_col) or "").strip()
            if len(text) < min_chars:
                continue

            report_id = str(row.get(id_col) or f"row_{idx}") if id_col else f"row_{idx}"
            label = None
            if label_col:
                raw = (row.get(label_col) or "").strip()
                label = raw if raw else None

            rows.append(ReportRecord(report_id=report_id, text=text, label=label))

            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise ValueError(
            f"No valid rows loaded from {csv_path}. "
            f"Check text column '{text_col}' and min_chars={min_chars}."
        )

    return rows


def _load_reports_from_jsonl(
    jsonl_path: Path,
    text_col: str,
    id_col: Optional[str],
    label_col: Optional[str],
    limit: Optional[int],
    min_chars: int,
) -> list[ReportRecord]:
    rows: list[ReportRecord] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            text = str(obj.get(text_col, "")).strip()
            if len(text) < min_chars:
                continue

            report_id = str(obj.get(id_col) or f"row_{idx}") if id_col else f"row_{idx}"
            label = None
            if label_col:
                raw = str(obj.get(label_col, "")).strip()
                label = raw if raw else None

            rows.append(ReportRecord(report_id=report_id, text=text, label=label))
            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise ValueError(
            f"No valid rows loaded from {jsonl_path}. "
            f"Check text column '{text_col}' and min_chars={min_chars}."
        )

    return rows


def _load_reports_from_txt_dir(
    reports_dir: Path,
    txt_glob: str,
    limit: Optional[int],
    min_chars: int,
) -> list[ReportRecord]:
    rows: list[ReportRecord] = []
    files = sorted(reports_dir.glob(txt_glob))
    if not files:
        raise ValueError(f"No files matched pattern '{txt_glob}' in {reports_dir}")

    for path in files:
        if not path.is_file():
            continue
        if path.name.endswith(".smoke.txt"):
            continue

        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < min_chars:
            continue

        rows.append(ReportRecord(report_id=path.stem, text=text, label=None))
        if limit is not None and len(rows) >= limit:
            break

    if not rows:
        raise ValueError(
            f"No valid text reports loaded from {reports_dir}. "
            f"Check glob='{txt_glob}' and min_chars={min_chars}."
        )

    return rows


def _token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _lexical_overlap(a: str, b: str) -> float:
    ta = _token_set(a)
    tb = _token_set(b)
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb) / len(ta | tb))


def _build_faiss_index(embeddings: np.ndarray) -> Any:
    _require_module(
        faiss,
        "faiss",
        "pip install faiss-cpu  # or install faiss-gpu in your CUDA env",
    )
    assert faiss is not None

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings.astype(np.float32)))
    return index


def _neighbors_without_self(
    index: Any,
    embeddings: np.ndarray,
    top_k: int,
) -> tuple[list[list[int]], list[list[float]]]:
    n = embeddings.shape[0]
    k_search = min(top_k + 1, n)
    scores, idx = index.search(embeddings, k_search)

    all_neighbors: list[list[int]] = []
    all_scores: list[list[float]] = []

    for i in range(n):
        neigh: list[int] = []
        neigh_scores: list[float] = []
        for j, s in zip(idx[i].tolist(), scores[i].tolist()):
            if j == i:
                continue
            neigh.append(int(j))
            neigh_scores.append(float(s))
            if len(neigh) == top_k:
                break
        all_neighbors.append(neigh)
        all_scores.append(neigh_scores)

    return all_neighbors, all_scores


def _self_retrieval_top1(index: Any, embeddings: np.ndarray) -> tuple[float, float]:
    scores, idx = index.search(embeddings, 1)
    expected = np.arange(embeddings.shape[0])
    actual = idx[:, 0]
    rate = float((actual == expected).mean())
    mean_score = float(scores[:, 0].mean())
    return rate, mean_score


def _majority_label_accuracy(
    labels: list[Optional[str]],
    neighbors: list[list[int]],
) -> tuple[Optional[float], int]:
    correct = 0
    total = 0

    for i, label in enumerate(labels):
        if not label:
            continue

        candidate_labels = [labels[j] for j in neighbors[i] if labels[j]]
        if not candidate_labels:
            continue

        pred = Counter(candidate_labels).most_common(1)[0][0]
        total += 1
        if pred == label:
            correct += 1

    if total == 0:
        return None, 0

    return float(correct / total), total


def _mean_lexical_overlap(
    records: list[ReportRecord],
    neighbors: list[list[int]],
    sample_size: int,
    seed: int,
) -> float:
    if not records:
        return 0.0

    n = len(records)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[: min(sample_size, n)]

    vals: list[float] = []
    for i in indices:
        q = records[i].text
        for j in neighbors[i]:
            vals.append(_lexical_overlap(q, records[j].text))

    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_relevance_samples(
    path: Path,
    records: list[ReportRecord],
    neighbors: list[list[int]],
    neighbor_scores: list[list[float]],
    num_queries: int,
    seed: int,
) -> None:
    n = len(records)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    picked = indices[: min(num_queries, n)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in picked:
            query = records[i]
            payload = {
                "query_id": query.report_id,
                "query_text": query.text,
                "query_label": query.label,
                "neighbors": [],
            }
            for rank, (j, score) in enumerate(zip(neighbors[i], neighbor_scores[i]), start=1):
                neighbor = records[j]
                payload["neighbors"].append(
                    {
                        "rank": rank,
                        "report_id": neighbor.report_id,
                        "score": score,
                        "label": neighbor.label,
                        "lexical_overlap": _lexical_overlap(query.text, neighbor.text),
                        "text": neighbor.text,
                    }
                )
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _normalize_label_key(raw: str) -> str:
    text = str(raw).strip()
    if not text:
        return ""
    base = Path(text).name.strip().lower()
    if base.endswith(".txt"):
        base = base[:-4]
    return base


def _parse_label_set(raw: str, sep: str) -> set[str]:
    items = [part.strip() for part in str(raw).split(sep) if part.strip()]
    labels = {x for x in items}
    if "No Finding" in labels and len(labels) > 1:
        labels.discard("No Finding")
    return labels


def _load_nih_labels(
    labels_csv: Path,
    id_col: str,
    labels_col: str,
    labels_sep: str,
) -> dict[str, set[str]]:
    table: dict[str, set[str]] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Label CSV has no header: {labels_csv}")
        if id_col not in reader.fieldnames:
            raise ValueError(
                f"labels_id_col '{id_col}' not found in {labels_csv}. "
                f"Available columns: {reader.fieldnames}"
            )
        if labels_col not in reader.fieldnames:
            raise ValueError(
                f"labels_col '{labels_col}' not found in {labels_csv}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            rid = _normalize_label_key(row.get(id_col, ""))
            if not rid:
                continue
            labels = _parse_label_set(row.get(labels_col, ""), labels_sep)
            if labels:
                table[rid] = labels

    if not table:
        raise ValueError(f"No valid labels parsed from {labels_csv}")
    return table


def _match_report_to_labels(report_id: str, label_table: dict[str, set[str]]) -> Optional[set[str]]:
    raw = str(report_id).strip()
    if not raw:
        return None
    candidates = [
        raw,
        f"{raw}.png",
        Path(raw).name,
        f"{Path(raw).name}.png",
    ]
    for cand in candidates:
        key = _normalize_label_key(cand)
        if key in label_table:
            return label_table[key]
    return None


def _compute_label_topk_metrics(
    report_ids: list[str],
    neighbors: list[list[int]],
    label_table: dict[str, set[str]],
    unmatched_sample_cap: int = 20,
) -> dict[str, Any]:
    labels_by_idx: list[Optional[set[str]]] = []
    unmatched: list[str] = []
    for rid in report_ids:
        lbl = _match_report_to_labels(rid, label_table)
        labels_by_idx.append(lbl)
        if lbl is None and len(unmatched) < unmatched_sample_cap:
            unmatched.append(rid)

    n = len(report_ids)
    matched = sum(1 for x in labels_by_idx if x is not None)
    coverage = float(matched / max(1, n))

    overlap_hits = 0
    overlap_total = 0
    jacc_vals: list[float] = []
    majority_correct = 0
    majority_total = 0

    for i in range(n):
        q = labels_by_idx[i]
        if not q:
            continue

        neigh_sets = [labels_by_idx[j] for j in neighbors[i] if labels_by_idx[j]]
        if not neigh_sets:
            continue

        overlap_total += 1
        if any(len(q & ns) > 0 for ns in neigh_sets):
            overlap_hits += 1

        for ns in neigh_sets:
            union = q | ns
            if not union:
                continue
            jacc_vals.append(float(len(q & ns) / len(union)))

        votes: Counter[str] = Counter()
        for ns in neigh_sets:
            for tag in ns:
                votes[tag] += 1
        if votes:
            pred = votes.most_common(1)[0][0]
            majority_total += 1
            if pred in q:
                majority_correct += 1

    topk_any = float(overlap_hits / overlap_total) if overlap_total else None
    topk_jacc = float(sum(jacc_vals) / len(jacc_vals)) if jacc_vals else None
    topk_majority = (
        float(majority_correct / majority_total) if majority_total else None
    )

    return {
        "label_join_coverage": coverage,
        "label_topk_any_overlap_rate": topk_any,
        "label_topk_mean_jaccard": topk_jacc,
        "label_majority_topk_accuracy": topk_majority,
        "label_majority_eval_count": majority_total,
        "label_unmatched_sample_ids": unmatched,
    }


def _default_reports_csv() -> Optional[Path]:
    candidates = [
        Path("./nih_train_with_findings.csv"),
        Path("/Users/ziauddin/Downloads/nih_train_with_findings.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def _default_reports_dir() -> Optional[Path]:
    candidates = [
        Path("/workspace/report_gen/4B/reports_final"),
        Path("./reports_final"),
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path.resolve()
    return None


def _compare_models_for_winner(results: list[RunMetrics], min_label_coverage: float) -> dict[str, Any]:
    ranked: list[tuple[float, float, RunMetrics, str]] = []

    for item in results:
        has_label_quality = (
            item.label_join_coverage is not None
            and item.label_join_coverage >= min_label_coverage
            and item.label_majority_topk_accuracy is not None
        )
        if has_label_quality:
            quality = item.label_majority_topk_accuracy
            quality_name = "label_majority_topk_accuracy"
        elif (
            item.label_join_coverage is not None
            and item.label_join_coverage >= min_label_coverage
            and item.label_topk_any_overlap_rate is not None
        ):
            quality = item.label_topk_any_overlap_rate
            quality_name = "label_topk_any_overlap_rate"
        elif item.majority_label_top5_accuracy is not None:
            quality = item.majority_label_top5_accuracy
            quality_name = "majority_label_top5_accuracy"
        else:
            quality = item.mean_lexical_overlap_top5
            quality_name = "mean_lexical_overlap_top5"

        ranked.append((quality, -item.encode_seconds, item, quality_name))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_quality, _, best, quality_name = ranked[0]

    return {
        "winner_model_id": best.model_id,
        "quality_metric_used": quality_name,
        "quality_score": best_quality,
        "winner_encode_seconds": best.encode_seconds,
        "ranking": [
            {
                "model_id": item.model_id,
                "quality_score": quality,
                "quality_metric_used": qname,
                "encode_seconds": item.encode_seconds,
                "label_join_coverage": item.label_join_coverage,
            }
            for quality, _, item, qname in ranked
        ],
        "tie_break_rule": "quality desc, then encode_seconds asc",
    }


def _preflight_models(
    models: list[str],
    device: str,
    pooling: Optional[str],
    max_length: Optional[int],
    trust_remote_code: bool,
    prefer_sentence_transformers: bool,
    fail_on_model_error: bool,
    preflight_text: str,
) -> tuple[list[str], list[dict[str, str]]]:
    print("\n[preflight] validating model load + tiny encode for all models")
    ready: list[str] = []
    failures: list[dict[str, str]] = []

    for model_id in models:
        print(f"[preflight] model={model_id}")
        encoder = None
        try:
            encoder = HFTextEncoder(
                model_id=model_id,
                device=device,
                pooling=pooling,
                max_length=max_length,
                trust_remote_code=trust_remote_code,
                prefer_sentence_transformers=prefer_sentence_transformers,
            )
            emb = encoder.encode([preflight_text], batch_size=1, normalize=True)
            if emb.ndim != 2 or emb.shape[0] != 1:
                raise RuntimeError(f"unexpected preflight embedding shape: {emb.shape}")
            if np.isnan(emb).any() or np.isinf(emb).any():
                raise RuntimeError("NaN/Inf detected during preflight encode")

            ready.append(model_id)
            print(
                f"[preflight] ok model={model_id} dim={int(emb.shape[1])} device={encoder.device}"
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            failures.append({"model_id": model_id, "error": msg})
            print(f"[preflight][error] model={model_id} failed: {msg}")
            if fail_on_model_error:
                raise
        finally:
            del encoder
            _maybe_clear_cuda_cache()

    return ready, failures


def _run_for_model(
    model_id: str,
    records: list[ReportRecord],
    output_root: Path,
    batch_size: int,
    device: str,
    pooling: Optional[str],
    max_length: Optional[int],
    trust_remote_code: bool,
    top_k: int,
    relevance_queries: int,
    seed: int,
    prefer_sentence_transformers: bool,
    label_table: Optional[dict[str, set[str]]] = None,
    fail_on_nan: bool = False,
) -> RunMetrics:
    model_dir = output_root / _slugify_model_id(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[run] model={model_id}")

    encoder = HFTextEncoder(
        model_id=model_id,
        device=device,
        pooling=pooling,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        prefer_sentence_transformers=prefer_sentence_transformers,
    )

    reports = [r.text for r in records]
    report_ids = [r.report_id for r in records]
    labels = [r.label for r in records]

    nan_retry_used = False
    t0 = time.perf_counter()
    embeddings = encoder.encode(reports, batch_size=batch_size, normalize=True)
    encode_seconds = time.perf_counter() - t0

    shape_ok = bool(embeddings.ndim == 2 and embeddings.shape[0] == len(records))
    if not shape_ok:
        raise RuntimeError(
            f"Embedding row mismatch for {model_id}: got {embeddings.shape[0]}, expected {len(records)}"
        )

    dim = int(embeddings.shape[1])
    speed = float(len(records) / max(encode_seconds, 1e-6))

    # Smoke checks
    nonfinite_initial = int(_nonfinite_row_indices(embeddings).size)
    has_nan_or_inf = nonfinite_initial > 0
    if (
        has_nan_or_inf
        and encoder.backend == "transformers"
        and str(encoder.device).startswith("cuda")
        and not encoder.force_fp32
    ):
        print(f"[warn] NaN/Inf for {model_id} with precision={encoder.precision_mode}; retrying on CUDA fp32")
        nan_retry_used = True
        _maybe_clear_cuda_cache()
        encoder = HFTextEncoder(
            model_id=model_id,
            device=device,
            pooling=pooling,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            prefer_sentence_transformers=prefer_sentence_transformers,
            force_fp32=True,
        )
        t0_retry = time.perf_counter()
        embeddings = encoder.encode(reports, batch_size=batch_size, normalize=True)
        retry_seconds = time.perf_counter() - t0_retry
        encode_seconds += retry_seconds
        has_nan_or_inf = bool(_nonfinite_row_indices(embeddings).size > 0)

    reencoded_rows = 0
    zero_filled_rows = 0
    if has_nan_or_inf:
        bad_idx = _nonfinite_row_indices(embeddings)
        print(f"[warn] {model_id}: non-finite rows detected={len(bad_idx)}; re-encoding rows individually")
        for idx in bad_idx.tolist():
            row = encoder.encode([reports[idx]], batch_size=1, normalize=True)
            row_vec = row[0].astype(np.float32)
            if np.isfinite(row_vec).all():
                embeddings[idx] = row_vec
                reencoded_rows += 1
            else:
                embeddings[idx] = np.zeros((embeddings.shape[1],), dtype=np.float32)
                zero_filled_rows += 1

        if _nonfinite_row_indices(embeddings).size > 0:
            # Final safety: replace any remaining invalid rows with zeros.
            still_bad = _nonfinite_row_indices(embeddings)
            embeddings[still_bad] = 0.0
            zero_filled_rows += int(still_bad.size)

        has_nan_or_inf = bool(_nonfinite_row_indices(embeddings).size > 0)
        if has_nan_or_inf and fail_on_nan:
            raise RuntimeError(f"NaN/Inf detected in embeddings for {model_id} after repair")

    sample_n = min(16, len(records))
    sample_reports = reports[:sample_n]
    emb_a = encoder.encode(sample_reports, batch_size=max(1, min(8, sample_n)), normalize=True)
    emb_b = encoder.encode(sample_reports, batch_size=max(1, min(3, sample_n)), normalize=True)
    batch_diff = float(np.max(np.abs(emb_a - emb_b))) if sample_n else 0.0
    batch_stable = bool(batch_diff < 1e-4)

    np.save(model_dir / "embeddings.npy", embeddings.astype(np.float32))
    _write_json(model_dir / "report_ids.json", report_ids)
    _write_json(
        model_dir / "model_meta.json",
        {
            "model_id": model_id,
            "embedding_dim": dim,
            "pooling": encoder.pooling,
            "backend": encoder.backend,
            "run_date_utc": _utc_now_iso(),
            "num_reports": len(records),
            "batch_size": batch_size,
            "max_length": encoder.max_length,
            "device": encoder.device,
            "precision_mode": encoder.precision_mode,
            "nan_retry_used": nan_retry_used,
        },
    )

    index = _build_faiss_index(embeddings)
    self_rate, self_score_mean = _self_retrieval_top1(index, embeddings)
    neighbors, neighbor_scores = _neighbors_without_self(index, embeddings, top_k=top_k)

    majority_acc, majority_n = _majority_label_accuracy(labels, neighbors)
    mean_overlap = _mean_lexical_overlap(
        records,
        neighbors,
        sample_size=min(200, len(records)),
        seed=seed,
    )
    label_metrics: dict[str, Any] = {}
    if label_table is not None:
        label_metrics = _compute_label_topk_metrics(
            report_ids=report_ids,
            neighbors=neighbors,
            label_table=label_table,
            unmatched_sample_cap=20,
        )

    _write_relevance_samples(
        model_dir / "relevance_samples.jsonl",
        records,
        neighbors,
        neighbor_scores,
        num_queries=relevance_queries,
        seed=seed,
    )

    metrics = RunMetrics(
        model_id=model_id,
        output_dir=str(model_dir),
        embedding_dim=dim,
        num_reports=len(records),
        shape_ok=shape_ok,
        nan_inf_ok=not has_nan_or_inf,
        encode_seconds=encode_seconds,
        reports_per_second=speed,
        self_retrieval_top1_rate=self_rate,
        self_retrieval_top1_score_mean=self_score_mean,
        batch_stability_max_abs_diff=batch_diff,
        batch_stable=batch_stable,
        mean_lexical_overlap_top5=mean_overlap,
        majority_label_top5_accuracy=majority_acc,
        majority_label_eval_count=majority_n,
        label_join_coverage=label_metrics.get("label_join_coverage"),
        label_topk_any_overlap_rate=label_metrics.get("label_topk_any_overlap_rate"),
        label_topk_mean_jaccard=label_metrics.get("label_topk_mean_jaccard"),
        label_majority_topk_accuracy=label_metrics.get("label_majority_topk_accuracy"),
        label_majority_eval_count=label_metrics.get("label_majority_eval_count", 0),
        label_unmatched_sample_ids=label_metrics.get("label_unmatched_sample_ids"),
        nan_retry_used=nan_retry_used,
        precision_mode=encoder.precision_mode,
        nonfinite_rows_initial=nonfinite_initial,
        nonfinite_rows_final=int(_nonfinite_row_indices(embeddings).size),
        nonfinite_rows_reencoded=reencoded_rows,
        nonfinite_rows_zero_filled=zero_filled_rows,
    )

    _write_json(model_dir / "metrics.json", metrics.__dict__)

    print(
        "[done] "
        f"dim={dim} reports={len(records)} speed={speed:.2f}/s "
        f"self_top1={self_rate:.3f} batch_stable={batch_stable}"
    )

    if majority_acc is not None:
        print(f"[metric] majority_label_top5_accuracy={majority_acc:.4f} (n={majority_n})")
    else:
        print(f"[metric] mean_lexical_overlap_top5={mean_overlap:.4f}")

    return metrics


def _load_metrics_file(path: Path) -> RunMetrics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunMetrics(**payload)


def _run_models_with_per_model_install(
    args: argparse.Namespace,
    input_path: Path,
    input_type: str,
    output_root: Path,
    num_reports: int,
) -> None:
    print("\n[mode] per-model dependency install enabled")
    python_bin = args.python_bin or sys.executable
    script_path = Path(__file__).resolve()

    run_results: list[RunMetrics] = []
    model_failures: list[dict[str, Any]] = []

    for model_id in args.models:
        install_reqs: list[str] = _dependency_profile_for_model(model_id)
        try:
            _install_dependencies_for_model(model_id=model_id, python_bin=python_bin)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            model_failures.append(
                {
                    "model_id": model_id,
                    "stage": "install",
                    "dependencies": install_reqs,
                    "error": msg,
                }
            )
            print(f"[error] model={model_id} install failed: {msg}")
            if args.fail_on_model_error:
                raise
            continue

        cmd = [
            python_bin,
            str(script_path),
            "--models",
            model_id,
            "--output-dir",
            str(output_root),
            "--limit",
            str(args.limit),
            "--min-chars",
            str(args.min_chars),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--top-k",
            str(args.top_k),
            "--relevance-queries",
            str(args.relevance_queries),
            "--seed",
            str(args.seed),
            "--min-label-coverage",
            str(args.min_label_coverage),
            "--no-install-per-model",
            "--no-preflight-models",
            "--fail-on-model-error",
        ]

        if args.pooling is not None:
            cmd.extend(["--pooling", args.pooling])
        if args.max_length is not None:
            cmd.extend(["--max-length", str(args.max_length)])
        if not args.trust_remote_code:
            cmd.append("--no-trust-remote-code")
        if args.prefer_sentence_transformers:
            cmd.append("--prefer-sentence-transformers")
        if args.fail_on_nan:
            cmd.append("--fail-on-nan")
        if args.labels_csv is not None:
            cmd.extend(
                [
                    "--labels-csv",
                    str(args.labels_csv),
                    "--labels-id-col",
                    args.labels_id_col,
                    "--labels-col",
                    args.labels_col,
                    "--labels-sep",
                    args.labels_sep,
                ]
            )

        if input_type == "csv":
            cmd.extend(["--reports-csv", str(input_path), "--text-col", args.text_col, "--id-col", args.id_col])
            if args.label_col:
                cmd.extend(["--label-col", args.label_col])
        elif input_type == "jsonl":
            cmd.extend(["--reports-jsonl", str(input_path), "--text-col", args.text_col, "--id-col", args.id_col])
            if args.label_col:
                cmd.extend(["--label-col", args.label_col])
        else:
            cmd.extend(["--reports-dir", str(input_path), "--txt-glob", args.txt_glob])

        print(f"[run-subprocess] model={model_id}")
        logs_dir = output_root / "worker_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{_slugify_model_id(model_id)}.log"
        proc = subprocess.run(cmd, capture_output=True, text=True)
        combined_log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        log_path.write_text(combined_log, encoding="utf-8")
        if proc.returncode != 0:
            tail = _tail_text(combined_log, max_lines=80)
            model_failures.append(
                {
                    "model_id": model_id,
                    "stage": "run",
                    "dependencies": install_reqs,
                    "error": f"worker exited with code {proc.returncode}",
                    "log_path": str(log_path),
                    "error_tail": tail,
                }
            )
            print(f"[error] worker failed model={model_id} log={log_path}")
            if tail:
                print(tail)
            if args.fail_on_model_error:
                raise SystemExit(proc.returncode)
            continue

        metrics_path = output_root / _slugify_model_id(model_id) / "metrics.json"
        if not metrics_path.exists():
            model_failures.append(
                {
                    "model_id": model_id,
                    "stage": "run",
                    "dependencies": install_reqs,
                    "error": f"metrics file missing: {metrics_path}",
                }
            )
            if args.fail_on_model_error:
                raise SystemExit(1)
            continue

        run_results.append(_load_metrics_file(metrics_path))
        _maybe_clear_cuda_cache()

    if not run_results:
        raise SystemExit("All models failed in install-per-model mode.")

    winner = _compare_models_for_winner(run_results, min_label_coverage=args.min_label_coverage)
    run_summary = {
        "run_date_utc": _utc_now_iso(),
        "input_path": str(input_path),
        "input_type": input_type,
        "num_reports": num_reports,
        "requested_models": args.models,
        "models_ran": [m.model_id for m in run_results],
        "install_per_model": True,
        "labels_csv": str(args.labels_csv) if args.labels_csv else None,
        "min_label_coverage": args.min_label_coverage,
        "fail_on_nan": bool(args.fail_on_nan),
        "models": [item.__dict__ for item in run_results],
        "model_failures": model_failures,
        "winner": winner,
        "selection_rule": "retrieval quality first, then speed",
    }
    _write_json(output_root / "run_summary.json", run_summary)
    print("\n[summary]")
    print(json.dumps(winner, indent=2))
    print(f"[saved] {output_root / 'run_summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and compare HF report embeddings for CXR retrieval."
    )

    parser.add_argument("--reports-csv", type=Path, default=None, help="CSV input with report text")
    parser.add_argument("--reports-jsonl", type=Path, default=None, help="JSONL input with report text")
    parser.add_argument("--reports-dir", type=Path, default=None, help="Directory with one report per .txt file")
    parser.add_argument("--txt-glob", type=str, default="*.txt", help="Glob for report files inside --reports-dir")
    parser.add_argument("--text-col", type=str, default="findings_snippet")
    parser.add_argument("--id-col", type=str, default="image_path")
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--min-chars", type=int, default=5)
    parser.add_argument("--limit", type=int, default=100)

    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--pooling", type=str, default=None, choices=["cls", "mean", "pooler"])
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--prefer-sentence-transformers", action="store_true")

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--relevance-queries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--labels-id-col", type=str, default="Image Index")
    parser.add_argument("--labels-col", type=str, default="Finding Labels")
    parser.add_argument("--labels-sep", type=str, default="|")
    parser.add_argument("--min-label-coverage", type=float, default=0.8)
    parser.add_argument(
        "--install-per-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Install dependency profile before each model and run models in subprocesses.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=None,
        help="Python executable used for per-model pip install and subprocess runs.",
    )
    parser.add_argument(
        "--preflight-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run model load + tiny encode checks before full embedding run.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run only preflight checks and exit without full embedding generation.",
    )
    parser.add_argument(
        "--preflight-text",
        type=str,
        default="No acute cardiopulmonary abnormality.",
        help="Short text used for per-model preflight encode.",
    )
    parser.add_argument(
        "--fail-on-model-error",
        action="store_true",
        help="Stop immediately if any model fails to load/run.",
    )
    parser.add_argument(
        "--fail-on-nan",
        action="store_true",
        help="Fail if non-finite embeddings remain after retry/repair.",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/cxr_embeddings"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_module(np, "numpy", "pip install numpy")

    if args.install_per_model and args.preflight_only:
        raise SystemExit("Use either --install-per-model or --preflight-only, not both.")

    selected_inputs = sum(
        bool(x) for x in [args.reports_csv, args.reports_jsonl, args.reports_dir]
    )
    if selected_inputs > 1:
        raise SystemExit("Use only one of --reports-csv, --reports-jsonl, or --reports-dir")

    input_type = ""
    if args.reports_csv:
        input_path = args.reports_csv.resolve()
        input_type = "csv"
    elif args.reports_jsonl:
        input_path = args.reports_jsonl.resolve()
        input_type = "jsonl"
    elif args.reports_dir:
        input_path = args.reports_dir.resolve()
        input_type = "txt_dir"
    else:
        inferred_dir = _default_reports_dir()
        if inferred_dir is not None:
            input_path = inferred_dir
            input_type = "txt_dir"
        else:
            inferred_csv = _default_reports_csv()
            if inferred_csv is None:
                raise SystemExit(
                    "No input specified. Pass --reports-dir, --reports-csv, or --reports-jsonl. "
                    "Tip: --reports-dir /workspace/report_gen/4B/reports_final"
                )
            input_path = inferred_csv
            input_type = "csv"

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    if not (0.0 <= args.min_label_coverage <= 1.0):
        raise SystemExit("--min-label-coverage must be between 0 and 1")

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if input_type == "txt_dir":
        records = _load_reports_from_txt_dir(
            reports_dir=input_path,
            txt_glob=args.txt_glob,
            limit=args.limit,
            min_chars=args.min_chars,
        )
    elif input_path.suffix.lower() == ".csv":
        records = _load_reports_from_csv(
            csv_path=input_path,
            text_col=args.text_col,
            id_col=args.id_col,
            label_col=args.label_col,
            limit=args.limit,
            min_chars=args.min_chars,
        )
    else:
        records = _load_reports_from_jsonl(
            jsonl_path=input_path,
            text_col=args.text_col,
            id_col=args.id_col,
            label_col=args.label_col,
            limit=args.limit,
            min_chars=args.min_chars,
        )

    print(f"[info] loaded_reports={len(records)} input={input_path}")
    print(f"[info] models={args.models}")
    label_table: Optional[dict[str, set[str]]] = None
    if args.labels_csv is not None:
        labels_path = args.labels_csv.resolve()
        if not labels_path.exists():
            raise SystemExit(f"Label file not found: {labels_path}")
        label_table = _load_nih_labels(
            labels_csv=labels_path,
            id_col=args.labels_id_col,
            labels_col=args.labels_col,
            labels_sep=args.labels_sep,
        )
        args.labels_csv = labels_path
        print(f"[info] labels_csv={labels_path} loaded_entries={len(label_table)}")

    if args.install_per_model:
        _run_models_with_per_model_install(
            args=args,
            input_path=input_path,
            input_type=input_type,
            output_root=output_root,
            num_reports=len(records),
        )
        return

    preflight_failures: list[dict[str, str]] = []
    models_to_run = list(args.models)
    if args.preflight_models:
        models_to_run, preflight_failures = _preflight_models(
            models=args.models,
            device=args.device,
            pooling=args.pooling,
            max_length=args.max_length,
            trust_remote_code=args.trust_remote_code,
            prefer_sentence_transformers=args.prefer_sentence_transformers,
            fail_on_model_error=args.fail_on_model_error,
            preflight_text=args.preflight_text,
        )
        if not models_to_run:
            raise SystemExit(
                "No model passed preflight. Fix dependency/runtime issues and rerun."
            )
        if args.preflight_only:
            preflight_summary = {
                "run_date_utc": _utc_now_iso(),
                "input_path": str(input_path),
                "input_type": input_type,
                "num_reports": len(records),
                "requested_models": args.models,
                "models_ready": models_to_run,
                "preflight_failures": preflight_failures,
            }
            _write_json(output_root / "preflight_summary.json", preflight_summary)
            print("\n[preflight-summary]")
            print(json.dumps(preflight_summary, indent=2))
            print(f"[saved] {output_root / 'preflight_summary.json'}")
            return
    elif args.preflight_only:
        raise SystemExit("--preflight-only requires --preflight-models")

    run_results: list[RunMetrics] = []
    model_failures: list[dict[str, str]] = list(preflight_failures)
    for model_id in models_to_run:
        try:
            run_results.append(
                _run_for_model(
                    model_id=model_id,
                    records=records,
                    output_root=output_root,
                    batch_size=args.batch_size,
                    device=args.device,
                    pooling=args.pooling,
                    max_length=args.max_length,
                    trust_remote_code=args.trust_remote_code,
                    top_k=args.top_k,
                    relevance_queries=args.relevance_queries,
                    seed=args.seed,
                    prefer_sentence_transformers=args.prefer_sentence_transformers,
                    label_table=label_table,
                    fail_on_nan=args.fail_on_nan,
                )
            )
        except Exception as exc:
            msg = str(exc)
            model_failures.append({"model_id": model_id, "error": msg})
            print(f"[error] model={model_id} failed: {msg}")
            if args.fail_on_model_error:
                raise

    if not run_results:
        raise SystemExit(
            "All models failed. Inspect logs above and try: "
            "`pip install -U 'transformers==4.57.6' 'accelerate==1.13.0' 'safetensors==0.7.0' "
            "'packaging==24.1' 'huggingface-hub==0.36.2' 'tokenizers==0.22.2'`."
        )

    winner = _compare_models_for_winner(run_results, min_label_coverage=args.min_label_coverage)

    run_summary = {
        "run_date_utc": _utc_now_iso(),
        "input_path": str(input_path),
        "input_type": input_type,
        "num_reports": len(records),
        "requested_models": args.models,
        "models_ran": [m.model_id for m in run_results],
        "preflight_models_enabled": bool(args.preflight_models),
        "labels_csv": str(args.labels_csv) if args.labels_csv else None,
        "min_label_coverage": args.min_label_coverage,
        "fail_on_nan": bool(args.fail_on_nan),
        "models": [item.__dict__ for item in run_results],
        "model_failures": model_failures,
        "winner": winner,
        "selection_rule": "retrieval quality first, then speed",
    }

    _write_json(output_root / "run_summary.json", run_summary)

    print("\n[summary]")
    print(json.dumps(winner, indent=2))
    print(f"[saved] {output_root / 'run_summary.json'}")


if __name__ == "__main__":
    main()

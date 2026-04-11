#!/usr/bin/env python3
"""Minimal local helpers for CXR Foundation image embedding export.

This module intentionally mirrors the official Google CXR Foundation local
Hugging Face flow for image inputs:

1. Convert each chest X-ray into a grayscale PNG-backed `tf.train.Example`.
2. Run `elixr-c-v2-pooled` to obtain ELIXR-C feature maps.
3. Run `pax-elixr-b-text` with zero text inputs to obtain image embeddings.

The preprocessing logic is adapted from the official open-source repository:
https://github.com/Google-Health/cxr-foundation
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


HF_REPO_ID = "google/cxr-foundation"
ELIXR_C_SUBDIR = "elixr-c-v2-pooled"
QFORMER_SUBDIR = "pax-elixr-b-text"
TEXT_INPUT_SHAPE = (1, 128)


def import_runtime_dependencies() -> tuple[Any, Any]:
    """Import TensorFlow runtime dependencies lazily.

    This keeps `--help` and syntax checks working even before TensorFlow is
    installed in the current environment.
    """

    try:
        import huggingface_hub
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'huggingface_hub'. Install CXR Foundation requirements with:\n"
            "  python -m pip install -r /workspace/scripts/requirements_cxr_foundation.txt"
        ) from exc

    try:
        import png  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'pypng'. Install CXR Foundation requirements with:\n"
            "  python -m pip install -r /workspace/scripts/requirements_cxr_foundation.txt"
        ) from exc

    try:
        import tensorflow as tf
        import tensorflow_text  # noqa: F401  # pylint: disable=unused-import
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing TensorFlow runtime dependencies. Install CXR Foundation requirements with:\n"
            "  python -m pip install -r /workspace/scripts/requirements_cxr_foundation.txt"
        ) from exc
    return tf, huggingface_hub


def _encode_png(array: np.ndarray) -> bytes:
    try:
        import png  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing 'pypng' dependency.") from exc

    if array.ndim != 2:
        raise ValueError(f"Array must be 2-D. Actual dimensions: {array.ndim}")
    if array.dtype.type not in {np.uint8, np.uint16}:
        raise ValueError(
            "Pixels must be uint8 or uint16. "
            f"Actual type: {array.dtype.name!r}"
        )
    writer = png.Writer(
        width=int(array.shape[1]),
        height=int(array.shape[0]),
        greyscale=True,
        bitdepth=8 * array.dtype.itemsize,
    )
    output = io.BytesIO()
    writer.write(output, array.tolist())
    return output.getvalue()


def _rescale_dynamic_range(image: np.ndarray) -> np.ndarray:
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(
            "Image pixels must be an integer type. "
            f"Actual type: {image.dtype.name!r}"
        )
    info = np.iinfo(image.dtype)
    if int(image.max()) == int(image.min()):
        return np.zeros_like(image, dtype=image.dtype)
    return np.interp(
        image,
        (image.min(), image.max()),
        (info.min, info.max),
    ).astype(info)


def _shift_to_unsigned(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint16 or image.dtype == np.uint8:
        return image
    if image.dtype == np.int16:
        image = image.astype(np.int32)
        return (image - np.min(image)).astype(np.uint16)
    if image.dtype == np.int8:
        image = image.astype(np.int16)
        return (image - np.min(image)).astype(np.uint8)
    if image.dtype == float:
        uint16_max = np.iinfo(np.uint16).max
        image = image - np.min(image)
        if np.max(image) > uint16_max:
            image = image * (uint16_max / np.max(image))
            image[image > uint16_max] = uint16_max
        return image.astype(np.uint16)
    raise ValueError(
        "Image pixels must be uint8, uint16, int8, int16, or float. "
        f"Actual type: {image.dtype.name!r}"
    )


def serialize_pil_image_to_tf_example(image: Image.Image) -> bytes:
    """Serialize a PIL image into the official PNG-backed tf.Example format."""

    image_array = np.asarray(image.convert("L"))
    pixel_array = _shift_to_unsigned(image_array)
    if pixel_array.dtype != np.uint8:
        pixel_array = _rescale_dynamic_range(pixel_array)
    png_bytes = _encode_png(pixel_array.astype(np.uint16))

    tf, _ = import_runtime_dependencies()
    image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[png_bytes]))
    format_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"png"]))
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/encoded": image_feature,
                "image/format": format_feature,
            }
        )
    )
    return example.SerializeToString()


def configure_tensorflow_memory_growth(tf: Any) -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception:
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            continue


def pool_token_embeddings(embeddings: np.ndarray, pooling: str) -> np.ndarray:
    if embeddings.ndim == 2:
        return embeddings.astype(np.float32, copy=False)
    if embeddings.ndim != 3:
        raise ValueError(f"Expected embeddings with 2 or 3 dims, found {embeddings.shape}")
    if pooling == "none":
        return embeddings.astype(np.float32, copy=False)
    if pooling == "avg":
        return embeddings.mean(axis=1).astype(np.float32, copy=False)
    if pooling == "cls":
        return embeddings[:, 0, :].astype(np.float32, copy=False)
    if pooling == "flatten":
        batch, tokens, dim = embeddings.shape
        return embeddings.reshape(batch, tokens * dim).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported token pooling mode: {pooling}")


class CxrFoundationImageEmbedder:
    """Local Hugging Face image embedder for CXR Foundation."""

    def __init__(
        self,
        *,
        model_dir: Path,
        hf_repo_id: str = HF_REPO_ID,
        hf_token: str | None = None,
    ) -> None:
        tf, huggingface_hub = import_runtime_dependencies()
        configure_tensorflow_memory_growth(tf)
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            huggingface_hub.snapshot_download(
                repo_id=hf_repo_id,
                local_dir=str(model_dir),
                allow_patterns=[f"{ELIXR_C_SUBDIR}/*", f"{QFORMER_SUBDIR}/*"],
                token=hf_token,
            )
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                "Failed to download CXR Foundation model files from Hugging Face.\n"
                "If the model terms were not previously accepted for this account, accept them on:\n"
                f"  https://huggingface.co/{hf_repo_id}\n"
                "Then set an access token in HF_TOKEN and rerun."
            ) from exc

        try:
            self._tf = tf
            self._elixr_c = tf.saved_model.load(str(model_dir / ELIXR_C_SUBDIR))
            self._qformer = tf.saved_model.load(str(model_dir / QFORMER_SUBDIR))
            self._elixr_c_infer = self._elixr_c.signatures["serving_default"]
            self._qformer_infer = self._qformer.signatures["serving_default"]
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                "Failed to load local CXR Foundation SavedModels from the downloaded snapshot."
            ) from exc

    def embed_serialized_examples(
        self,
        serialized_examples: list[bytes],
        *,
        embedding_kind: str,
        token_pooling: str,
    ) -> np.ndarray:
        if not serialized_examples:
            raise ValueError("serialized_examples is empty")

        batch_size = len(serialized_examples)
        tf = self._tf
        elixr_c_output = self._elixr_c_infer(
            input_example=tf.constant(serialized_examples)
        )
        qformer_output = self._qformer_infer(
            image_feature=elixr_c_output["feature_maps_0"],
            ids=np.zeros((batch_size, *TEXT_INPUT_SHAPE), dtype=np.int32),
            paddings=np.zeros((batch_size, *TEXT_INPUT_SHAPE), dtype=np.float32),
        )

        if embedding_kind == "general":
            raw_embeddings = qformer_output["img_emb"].numpy()
        elif embedding_kind == "contrastive":
            raw_embeddings = qformer_output["all_contrastive_img_emb"].numpy()
        else:
            raise ValueError(f"Unsupported embedding kind: {embedding_kind}")
        return pool_token_embeddings(np.asarray(raw_embeddings), token_pooling)


def resolve_hf_token(env_var: str = "HF_TOKEN") -> str | None:
    token = os.environ.get(env_var)
    if token:
        token = token.strip()
    return token or None

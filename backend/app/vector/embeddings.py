"""Process-wide cached MiniLM and CLIP embedding models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"
MINILM_DIMENSION = 384
CLIP_DIMENSION = 512


@lru_cache(maxsize=1)
def minilm() -> SentenceTransformer:
    try:
        return SentenceTransformer(MINILM_MODEL, device="cpu", local_files_only=True)
    except OSError:
        return SentenceTransformer(MINILM_MODEL, device="cpu")


@lru_cache(maxsize=1)
def clip_components() -> tuple[CLIPModel, CLIPProcessor]:
    try:
        model = CLIPModel.from_pretrained(CLIP_MODEL, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL, local_files_only=True)
    except OSError:
        model = CLIPModel.from_pretrained(CLIP_MODEL)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.eval()
    return model, processor


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = minilm().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    if vectors.shape[1] != MINILM_DIMENSION:
        raise RuntimeError(f"unexpected MiniLM dimension: {vectors.shape[1]}")
    return vectors.astype(np.float32).tolist()


def embed_clip_images(paths: list[Path]) -> list[list[float]]:
    model, processor = clip_components()
    vectors: list[list[float]] = []
    for offset in range(0, len(paths), 4):
        images = []
        for path in paths[offset:offset + 4]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        vectors.extend(features.cpu().numpy().astype(np.float32).tolist())
    if vectors and len(vectors[0]) != CLIP_DIMENSION:
        raise RuntimeError(f"unexpected CLIP dimension: {len(vectors[0])}")
    return vectors


def embed_clip_query(text: str) -> list[float]:
    model, processor = clip_components()
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.inference_mode():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    vector = features[0].cpu().numpy().astype(np.float32).tolist()
    if len(vector) != CLIP_DIMENSION:
        raise RuntimeError(f"unexpected CLIP dimension: {len(vector)}")
    return vector

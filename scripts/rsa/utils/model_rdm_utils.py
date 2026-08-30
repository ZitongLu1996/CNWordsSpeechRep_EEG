"""Utilities for building, validating, and saving model RDM bundles."""

import csv
import json
from pathlib import Path
import traceback

import h5py
import numpy as np

from utils.config import TENCENT_EMBEDDING_PATH
from utils.rdm_utils import save_rdm_heatmap, validate_square_rdm


PINYIN_BY_WORD = {
    "我": "wo3",
    "你": "ni3",
    "吃": "chi1",
    "喝": "he1",
    "好": "hao3",
    "不": "bu4",
    "冷": "leng3",
    "热": "re4",
    "左": "zuo3",
    "右": "you4",
}

# Explicit decomposition avoids relying on a pronunciation package or model.
PINYIN_PARTS_BY_WORD = {
    "我": ("w", "o", "3"),
    "你": ("n", "i", "3"),
    "吃": ("ch", "i", "1"),
    "喝": ("h", "e", "1"),
    "好": ("h", "ao", "3"),
    "不": ("b", "u", "4"),
    "冷": ("l", "eng", "3"),
    "热": ("r", "e", "4"),
    "左": ("z", "uo", "3"),
    "右": ("y", "ou", "4"),
}


def compute_cosine_rdm(features):
    """Compute a finite, symmetric cosine-distance RDM from item features."""
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError(
            "features must have shape n_items x n_features with at least two items; "
            f"got {matrix.shape}."
        )
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"features must be numeric; got dtype {matrix.dtype}.")
    matrix = matrix.astype(float, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("features contain NaN or infinite values.")
    norms = np.linalg.norm(matrix, axis=1)
    zero_rows = np.flatnonzero(np.isclose(norms, 0.0))
    if zero_rows.size:
        raise ValueError(f"features contain all-zero item vectors at rows {zero_rows.tolist()}.")
    normalized = matrix / norms[:, np.newaxis]
    rdm = 1.0 - normalized @ normalized.T
    rdm = (rdm + rdm.T) / 2.0
    np.fill_diagonal(rdm, 0.0)
    rdm[np.abs(rdm) < 1e-15] = 0.0
    validate_square_rdm(rdm, "cosine_rdm")
    if rdm.min() < -1e-10 or rdm.max() > 2.0 + 1e-10:
        raise ValueError(f"Cosine distances outside [0, 2]: min={rdm.min()}, max={rdm.max()}.")
    return rdm


def build_phonology_features(words):
    """Build interpretable initial/final/tone concatenated one-hot features."""
    order = [str(word) for word in words]
    missing = [word for word in order if word not in PINYIN_PARTS_BY_WORD]
    if missing:
        raise ValueError(f"No manually defined pinyin decomposition for words: {missing}.")
    parts = [PINYIN_PARTS_BY_WORD[word] for word in order]
    initials = sorted({part[0] for part in parts})
    finals = sorted({part[1] for part in parts})
    tones = sorted({part[2] for part in parts})
    feature_names = (
        [f"initial:{value}" for value in initials]
        + [f"final:{value}" for value in finals]
        + [f"tone:{value}" for value in tones]
    )
    matrix = np.zeros((len(order), len(feature_names)), dtype=float)
    initial_offset = 0
    final_offset = len(initials)
    tone_offset = len(initials) + len(finals)
    for row, (initial, final, tone) in enumerate(parts):
        matrix[row, initial_offset + initials.index(initial)] = 1.0
        matrix[row, final_offset + finals.index(final)] = 1.0
        matrix[row, tone_offset + tones.index(tone)] = 1.0
    metadata = {
        "pinyin": {word: PINYIN_BY_WORD[word] for word in order},
        "pinyin_parts": {
            word: {"initial": part[0], "final": part[1], "tone": part[2]}
            for word, part in zip(order, parts)
        },
        "initial_categories": initials,
        "final_categories": finals,
        "tone_categories": tones,
    }
    return matrix, feature_names, metadata


def build_tencent_embeddings(
    words,
    embedding_path=None,
    text2vec_model_name="w2v-light-tencent-chinese",
):
    """Load official Tencent TXT vectors with gensim, returning skip metadata on failure.

    ``text2vec_model_name`` is retained only for API compatibility and provenance;
    this pipeline intentionally does not download or use text2vec. The official
    local UTF-8 text file is always loaded with ``binary=False``.
    """
    order = [str(word) for word in words]
    path = Path(embedding_path or TENCENT_EMBEDDING_PATH)
    metadata = {
        "model_name": "semantic_tencent",
        "embedding_source": "Tencent AI Lab Embedding Corpus for Chinese Words and Phrases",
        "embedding_file": str(path),
        "backend": "gensim.KeyedVectors.load_word2vec_format",
        "binary": False,
        "encoding": "UTF-8",
        "text2vec_model_name_not_used": text2vec_model_name,
        "word_order": order,
        "missing_words": [],
        "status": "pending",
    }
    try:
        if not path.is_file():
            raise FileNotFoundError(f"Tencent embedding file not found: {path}")
        from gensim.models import KeyedVectors

        # Required official TXT loading route; do not change to binary=True.
        keyed_vectors = KeyedVectors.load_word2vec_format(path, binary=False)
        missing = [word for word in order if word not in keyed_vectors.key_to_index]
        metadata["missing_words"] = missing
        metadata["vocabulary_size"] = int(len(keyed_vectors))
        metadata["embedding_dim"] = int(keyed_vectors.vector_size)
        if missing:
            metadata["status"] = "skipped"
            metadata["error"] = f"Missing required words: {missing}"
            return None, metadata
        embeddings = np.vstack([keyed_vectors[word] for word in order]).astype(float)
        if embeddings.shape[0] != len(order):
            raise ValueError(f"Unexpected embedding matrix shape: {embeddings.shape}.")
        if not np.isfinite(embeddings).all():
            raise ValueError("Tencent embeddings contain NaN or infinite values.")
        zero_words = [word for word, vector in zip(order, embeddings)
                      if np.isclose(np.linalg.norm(vector), 0.0)]
        if zero_words:
            raise ValueError(f"Tencent embeddings contain all-zero vectors: {zero_words}.")
        metadata["status"] = "ok"
        return embeddings, metadata
    except Exception as exc:
        metadata.update({
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        return None, metadata


def _json_ready(value):
    """Convert nested NumPy and Path objects to JSON-serializable values."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _save_feature_csv(path, features, words, feature_names):
    """Save labeled interpretable features as UTF-8 CSV."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["word", *feature_names])
        for word, row in zip(words, features):
            writer.writerow([word, *row.tolist()])


def save_model_rdm_bundle(
    save_prefix,
    rdm,
    labels,
    words,
    metadata,
    features=None,
    feature_names=None,
):
    """Save one validated model RDM as NPY, HDF5, JSON, PNG, and features."""
    prefix = Path(save_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rdm_array = validate_square_rdm(rdm, f"{prefix.name}_rdm")
    label_order = [int(label) for label in labels]
    word_order = [str(word) for word in words]
    n_items = rdm_array.shape[0]
    if len(label_order) != n_items or len(word_order) != n_items:
        raise ValueError("labels and words must match the RDM condition dimension.")
    if len(set(label_order)) != n_items or len(set(word_order)) != n_items:
        raise ValueError("labels and words must be unique and ordered explicitly.")
    if not np.isfinite(rdm_array).all():
        raise ValueError("rdm contains NaN or infinite values.")
    if not np.allclose(rdm_array, rdm_array.T, rtol=1e-7, atol=1e-8):
        raise ValueError("rdm must be symmetric.")
    if not np.allclose(np.diag(rdm_array), 0.0, atol=1e-8):
        raise ValueError("rdm diagonal must be zero.")

    paths = {
        "rdm_npy": prefix.with_name(prefix.name + "_rdm.npy"),
        "rdm_h5": prefix.with_name(prefix.name + "_rdm.h5"),
        "rdm_png": prefix.with_name(prefix.name + "_rdm.png"),
        "metadata_json": prefix.with_name(prefix.name + "_metadata.json"),
    }
    bundle_metadata = dict(metadata)
    bundle_metadata["word_order"] = word_order
    bundle_metadata["label_order"] = label_order
    bundle_metadata["rdm_shape"] = list(rdm_array.shape)
    bundle_metadata["n_nan"] = int(np.isnan(rdm_array).sum())
    metadata_json = json.dumps(_json_ready(bundle_metadata), ensure_ascii=False, indent=2)

    np.save(paths["rdm_npy"], rdm_array)
    utf8 = h5py.string_dtype(encoding="utf-8")
    with h5py.File(paths["rdm_h5"], "w") as handle:
        handle.create_dataset("rdm", data=rdm_array)
        handle.create_dataset("label_order", data=np.asarray(label_order, dtype=int))
        handle.create_dataset("word_order", data=np.asarray(word_order, dtype=object), dtype=utf8)
        handle.create_dataset("metadata_json", data=metadata_json, dtype=utf8)
    paths["metadata_json"].write_text(metadata_json + "\n", encoding="utf-8")
    save_rdm_heatmap(rdm_array, word_order, paths["rdm_png"], title=bundle_metadata.get("model_name"))

    if features is not None:
        feature_array = np.asarray(features)
        if feature_array.ndim != 2 or feature_array.shape[0] != n_items:
            raise ValueError(f"features must have {n_items} rows; got {feature_array.shape}.")
        if not np.isfinite(feature_array).all():
            raise ValueError("features contain NaN or infinite values.")
        if feature_names is not None:
            names = [str(name) for name in feature_names]
            if len(names) != feature_array.shape[1]:
                raise ValueError("feature_names length does not match feature columns.")
            paths["features"] = prefix.with_name(prefix.name + "_features.csv")
            _save_feature_csv(paths["features"], feature_array, word_order, names)
        else:
            paths["features"] = prefix.with_name(prefix.name + "_features.npy")
            np.save(paths["features"], feature_array)
    return paths


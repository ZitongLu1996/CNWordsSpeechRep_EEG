"""Utilities for experimental-image AlexNet visual model RDMs."""

import importlib.metadata
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from utils.model_rdm_utils import compute_cosine_rdm, save_model_rdm_bundle


class VisualModelUnavailableError(RuntimeError):
    """Raised when required local CNN software or pretrained weights are absent."""


def find_character_image_paths(image_dir, words):
    """Map fixed-order words to unique numbered experimental image files."""
    directory = Path(image_dir)
    order = [str(word) for word in words]
    if not directory.is_dir():
        raise FileNotFoundError(f"Experimental character image directory not found: {directory}")
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    numeric_pngs = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in allowed_suffixes and path.stem.isdigit():
            numeric_pngs.setdefault(int(path.stem), []).append(path)
    missing = [index for index in range(1, len(order) + 1) if index not in numeric_pngs]
    if missing:
        raise FileNotFoundError(
            "Missing numbered experimental character images for the fixed word order: "
            + ", ".join(str(index) for index in missing)
        )
    ambiguous = {
        index: [str(path) for path in paths]
        for index, paths in numeric_pngs.items()
        if 1 <= index <= len(order) and len(paths) != 1
    }
    if ambiguous:
        raise ValueError(f"Ambiguous numeric image files map to the same trigger: {ambiguous}.")
    paths = [numeric_pngs[index][0] for index in range(1, len(order) + 1)]
    if len(set(path.resolve() for path in paths)) != len(order):
        raise ValueError("Character image mapping contains duplicate paths.")
    return paths


def inspect_character_images(image_paths):
    """Open every source image and return auditable path/size/mode metadata."""
    records = []
    for path_like in image_paths:
        path = Path(path_like)
        try:
            with Image.open(path) as image:
                image.load()
                records.append({
                    "path": str(path.resolve()),
                    "original_size": [int(image.width), int(image.height)],
                    "original_mode": image.mode,
                    "read_success": True,
                })
        except Exception as exc:
            raise ValueError(f"Failed to open experimental image {path}: {type(exc).__name__}: {exc}") from exc
    return records


def load_and_preprocess_images(image_paths):
    """Load real images, convert RGB, resize, normalize, and return a Torch batch."""
    image_metadata = inspect_character_images(image_paths)
    try:
        import torch
        from torchvision import transforms
    except Exception as exc:
        raise VisualModelUnavailableError(
            "PyTorch/torchvision is unavailable; prepare both packages and local "
            f"AlexNet pretrained weights before rerunning. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    tensors = []
    for path_like in image_paths:
        with Image.open(path_like) as image:
            tensors.append(preprocessing(image.convert("RGB")))
    batch = torch.stack(tensors, dim=0)
    expected_shape = (len(image_paths), 3, 224, 224)
    if tuple(batch.shape) != expected_shape:
        raise ValueError(f"Preprocessed batch shape {tuple(batch.shape)} != {expected_shape}.")
    if not bool(torch.isfinite(batch).all()):
        raise ValueError("Preprocessed image batch contains NaN or infinite values.")
    metadata = {
        "images": image_metadata,
        "converted_mode": "RGB",
        "resize": [224, 224],
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "batch_shape": list(batch.shape),
    }
    return batch, metadata


def _alexnet_checkpoint_path(torch, weights):
    """Resolve the official torchvision checkpoint path under configured TORCH_HOME."""
    filename = Path(urlparse(weights.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def get_alexnet_conv_features(images, layer_index=3):
    """Extract flattened official pretrained AlexNet conv2 maps, allowing download."""
    try:
        import torch
        import torchvision
    except Exception as exc:
        raise VisualModelUnavailableError(
            "PyTorch/torchvision is unavailable; no pretrained model was loaded. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(layer_index, int):
        raise TypeError("layer_index must be an integer index into alexnet.features.")
    weights_api_used = ""
    weights_name = ""
    checkpoint = None
    try:
        from torchvision.models import AlexNet_Weights, alexnet

        weights = AlexNet_Weights.DEFAULT
        weights_api_used = "torchvision.models.alexnet(weights=AlexNet_Weights.DEFAULT)"
        weights_name = str(weights)
        checkpoint = _alexnet_checkpoint_path(torch, weights)
        model = alexnet(weights=weights)
    except (ImportError, AttributeError):
        # Compatibility only: older torchvision releases lack the weights enum.
        weights_api_used = "torchvision.models.alexnet(pretrained=True)"
        weights_name = "pretrained=True"
        model = torchvision.models.alexnet(pretrained=True)
        candidates = sorted((Path(torch.hub.get_dir()) / "checkpoints").glob("alexnet*.pth"))
        checkpoint = candidates[-1] if candidates else None

    model.eval()
    if layer_index < 0 or layer_index >= len(model.features):
        raise IndexError(
            f"layer_index={layer_index} outside alexnet.features length {len(model.features)}."
        )
    layer = model.features[layer_index]
    activations = {}

    def capture(_module, _inputs, output):
        """Store the selected layer output for this forward pass."""
        activations["value"] = output.detach().cpu()

    hook = layer.register_forward_hook(capture)
    try:
        with torch.no_grad():
            model(images.cpu())
    finally:
        hook.remove()
    if "value" not in activations:
        raise RuntimeError("AlexNet hook did not capture an activation tensor.")
    feature_map = activations["value"]
    features = feature_map.reshape(feature_map.shape[0], -1).numpy().astype(float)
    if features.ndim != 2 or features.shape[0] != images.shape[0]:
        raise ValueError(f"Unexpected flattened feature shape: {features.shape}.")
    if not np.isfinite(features).all():
        raise ValueError("AlexNet features contain NaN or infinite values.")
    zero_rows = np.flatnonzero(np.isclose(np.linalg.norm(features, axis=1), 0.0))
    if zero_rows.size:
        raise ValueError(f"AlexNet produced all-zero features for rows {zero_rows.tolist()}.")
    metadata = {
        "cnn_backend": "torchvision.models.alexnet",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "weights_api_used": weights_api_used,
        "pretrained_weights": weights_name,
        "pretrained_weights_status": "downloaded_or_loaded_successfully",
        "weights_checkpoint": str(checkpoint.resolve()) if checkpoint and checkpoint.is_file() else None,
        "torch_home": os.environ.get("TORCH_HOME"),
        "torch_hub_dir": torch.hub.get_dir(),
        "download_allowed": True,
        "random_init": False,
        "features_structure": [f"{index}: {module}" for index, module in enumerate(model.features)],
        "hook_layer_index": layer_index,
        "hook_layer_name": type(layer).__name__,
        "feature_map_shape": list(feature_map.shape),
        "feature_processing": "flatten",
        "feature_dim": int(features.shape[1]),
    }
    return features, metadata


def save_visual_rdm_bundle(save_prefix, rdm, labels, words, metadata, features):
    """Save the AlexNet visual RDM bundle using shared validated serializers."""
    return save_model_rdm_bundle(
        save_prefix=save_prefix,
        rdm=rdm,
        labels=labels,
        words=words,
        metadata=metadata,
        features=features,
        feature_names=None,
    )


def save_skipped_visual_metadata(path, metadata):
    """Persist image audit and model-unavailability details for a skipped run."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "VisualModelUnavailableError",
    "compute_cosine_rdm",
    "find_character_image_paths",
    "get_alexnet_conv_features",
    "inspect_character_images",
    "load_and_preprocess_images",
    "save_skipped_visual_metadata",
    "save_visual_rdm_bundle",
]

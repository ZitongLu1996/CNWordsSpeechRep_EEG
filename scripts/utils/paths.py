"""Repository-relative paths with CLI and optional YAML overrides."""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when using a config file") from exc
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def add_path_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-rdm-root", type=Path)
    return parser


def resolve_paths(args=None):
    values = _yaml(args.config) if args is not None else _yaml(PROJECT_ROOT / "config" / "config.yaml")
    paths = values.get("paths", {})
    def choose(cli_value, key, default):
        value = cli_value if cli_value is not None else paths.get(key, default)
        path = Path(value).expanduser()
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    return {
        "project_root": PROJECT_ROOT,
        "data_root": choose(getattr(args, "data_root", None), "data_root", "data"),
        "output_root": choose(getattr(args, "output_root", None), "output_root", "outputs"),
        "model_rdm_root": choose(getattr(args, "model_rdm_root", None), "model_rdm_root", "model_rdms"),
    }

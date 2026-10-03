"""Package-owned checkpoint metadata and validation for the strict TOML registry.

`config/checkpoints.toml` is a smaller, stricter, sha256-verified sibling registry
(21 models, schema.version=1) layered on top of the legacy `data/melband_models.json`
(backing `model_registry.py`'s 99-entry `MODEL_REGISTRY`, this package's original and
still-primary registry): every model here is cross-referenced by slug against that
JSON registry and additionally carries per-artifact sha256 digests and provenance,
which the JSON registry does not. Callers (`download.py`, `clean_api.py`) look a
model up here for stricter metadata when present and fall back to JSON-only
handling otherwise -- this module does not replace the JSON registry, and being
absent here is not itself an error.
`load_checkpoints` validates the TOML eagerly and raises `ValueError` for any schema
violation: missing/wrong schema version, a malformed `models` table, a model missing
its `artifacts` list, or an artifact with a missing/malformed URL or SHA-256.
`checkpoint_metadata` raises `KeyError` (not `ValueError`) for a model name absent
from this TOML registry -- a distinct failure mode from a malformed file.

Reads: config/checkpoints.toml (via tomllib or tomli on Python 3.10)
"""
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def checkpoint_config_path() -> Path:
    return Path(__file__).with_name("config") / "checkpoints.toml"


def load_checkpoints(path=None) -> dict:
    path = Path(path) if path else checkpoint_config_path()
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid checkpoint config: {path}") from exc
    if data.get("schema", {}).get("version") != 1 or not isinstance(data.get("models"), dict):
        raise ValueError("checkpoint config must define schema.version=1 and models")
    for key, model in data["models"].items():
        if not isinstance(model, dict) or not isinstance(model.get("artifacts"), list):
            raise ValueError(f"invalid checkpoint metadata for {key}")
        for artifact in model["artifacts"]:
            if not isinstance(artifact, dict) or not str(artifact.get("url", "")).startswith("https://"):
                raise ValueError(f"invalid artifact URL for {key}")
            digest = artifact.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest.lower())
            ):
                raise ValueError(f"invalid artifact SHA-256 for {key}")
    return data


def checkpoint_metadata(model: str) -> dict:
    try:
        return dict(load_checkpoints()["models"][model])
    except KeyError as exc:
        raise KeyError(f"unknown checkpoint model: {model}") from exc

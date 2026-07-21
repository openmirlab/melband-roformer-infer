"""Additive lifecycle facade for Mel-Band Roformer inference with exact output manifests.

This module gives Python callers a package-owned session/separator API on top of
the CLI-oriented inference flow: load once, infer many times, then release GPU
memory explicitly. Session inference now returns the exact files run_folder()
wrote, so callers do not have to reconstruct output paths from conventions.

Reads: .checkpoints (checkpoint_metadata), .download (ensure_model_assets,
get_file_hash), .inference (SafeLoaderWithTuple, run_folder), .utils
(get_model_from_config), yaml, ml_collections, torch
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from ml_collections import ConfigDict

from .checkpoints import checkpoint_metadata

if TYPE_CHECKING:
    from .inference import OutputManifest


class MelBandRoformerSession:
    def __init__(
        self,
        *,
        model_name: str = "melband-roformer-kim-vocals",
        model: Any = None,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        config: ConfigDict | None = None,
        models_dir: str | Path | None = None,
        device: Any = None,
        progress: bool = True,
        checkpoint_url: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model
        self.model_path = Path(model_path) if model_path else None
        self.config_path = Path(config_path) if config_path else None
        self.models_dir = models_dir
        self.device = device
        self.progress = progress
        self.checkpoint_url = checkpoint_url
        self.checkpoint_sha256 = checkpoint_sha256
        self._config = config
        self._status = "ready" if model is not None else "new"

    @property
    def status(self) -> str:
        return self._status

    def _metadata(self):
        try:
            return checkpoint_metadata(self.model_name)
        except KeyError:
            return {"artifacts": []}

    def load(self) -> MelBandRoformerSession:
        if self._status == "ready":
            return self
        if self._status == "closed":
            raise RuntimeError("cannot load a closed MelBandRoformerSession")
        self._status = "loading"
        try:
            from .download import ensure_model_assets, get_file_hash
            from .inference import SafeLoaderWithTuple, _resolve_device
            from .utils import get_model_from_config

            if self.model_path is None or self.config_path is None:
                self.model_path, self.config_path = ensure_model_assets(
                    self.model_name,
                    models_dir=self.models_dir,
                    checkpoint_url=self.checkpoint_url,
                    checkpoint_sha256=self.checkpoint_sha256,
                )

            with self.config_path.open() as handle:
                self._config = ConfigDict(yaml.load(handle, Loader=SafeLoaderWithTuple))
            self._model = get_model_from_config("mel_band_roformer", self._config)

            import torch

            artifacts = self._metadata().get("artifacts", [])
            expected = self.checkpoint_sha256 or next(
                (a["sha256"] for a in artifacts if a.get("kind") == "checkpoint"),
                None,
            )
            if expected and get_file_hash(self.model_path).lower() != expected.lower():
                raise ValueError(f"checkpoint SHA-256 mismatch for {self.model_path}")

            self._model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            target = _resolve_device(self.device)
            self._model = self._model.to(target).eval()
            self.device = target
            self._status = "ready"
            return self
        except Exception:
            self._status = "failed"
            raise

    def infer(
        self,
        input_folder: str | Path,
        *,
        store_dir: str | Path = "outputs",
        verbose: bool = False,
    ) -> "OutputManifest":
        if self._status != "ready" or self._model is None:
            raise RuntimeError(
                "MelBandRoformerSession must be ready; call load() before infer()"
            )
        from .inference import run_folder
        from argparse import Namespace

        return run_folder(
            self._model,
            Namespace(input_folder=Path(input_folder), store_dir=Path(store_dir)),
            self._config,
            self.device,
            verbose=verbose,
        )

    def release(self) -> MelBandRoformerSession:
        if self._status == "closed":
            return self
        if self._model is not None and hasattr(self._model, "cpu"):
            self._model.cpu()
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self._status = "released"
        return self

    def close(self) -> MelBandRoformerSession:
        if self._status != "closed":
            self.release()
            self._status = "closed"
        return self

    def cache_info(self) -> dict[str, Any]:
        if self.model_path is not None:
            checkpoint_path = self.model_path
            config_path = self.config_path
        else:
            from .download import resolve_model_asset_paths

            checkpoint_path, config_path = resolve_model_asset_paths(
                self.model_name,
                self.models_dir,
                checkpoint_url=self.checkpoint_url,
            )
        return {
            "model": self.model_name,
            "status": self._status,
            "model_loaded": self._model is not None,
            "models_dir": str(self.models_dir) if self.models_dir else None,
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path) if config_path else None,
            "checkpoint_url": self.checkpoint_url
            or next(
                (
                    artifact["url"]
                    for artifact in self._metadata().get("artifacts", [])
                    if artifact.get("kind") == "checkpoint"
                ),
                None,
            ),
            "cached": checkpoint_path.is_file() and bool(config_path and config_path.is_file()),
        }

    def __enter__(self) -> MelBandRoformerSession:
        return self.load()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


class MelBandRoformerSeparator:
    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
        self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = MelBandRoformerSession(**self._kwargs).load()
        return self._session

    def __call__(
        self,
        input_folder: str | Path,
        *,
        store_dir: str | Path = "outputs",
        verbose: bool = False,
    ) -> "OutputManifest":
        return self.session.infer(input_folder, store_dir=store_dir, verbose=verbose)


def separate_folder(input_folder, **kwargs):
    allowed = {
        key: value
        for key, value in kwargs.items()
        if key
        in {
            "model_name",
            "model_path",
            "config_path",
            "models_dir",
            "device",
            "progress",
            "checkpoint_url",
            "checkpoint_sha256",
        }
    }
    return MelBandRoformerSeparator(**allowed).session.infer(
        input_folder,
        store_dir=kwargs.get("store_dir", "outputs"),
    )

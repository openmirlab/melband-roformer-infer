"""Public package surface -- re-exports the model, registry, and CLI entry points.

Thin barrel: no logic of its own, just the stable import path (`from mel_band_roformer
import ...`) that callers, tests, and the `melband-roformer-infer` /
`melband-roformer-download` console scripts in pyproject.toml depend on.

Reads: .mel_band_roformer, .model_registry, .utils, .inference, .download
"""

from .mel_band_roformer import MelBandRoformer  # noqa: F401
from .model_registry import MODEL_REGISTRY, MelBandModel, DEFAULT_MODEL  # noqa: F401
from .utils import demix_track, get_model_from_config  # noqa: F401
from .inference import main as inference_main  # noqa: F401
from .download import main as download_main  # noqa: F401

__all__ = [
    "MelBandRoformer",
    "MelBandModel",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL",
    "get_model_from_config",
    "demix_track",
    "inference_main",
    "download_main",
]

__version__ = "0.1.2"

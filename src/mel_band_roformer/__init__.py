"""Mel-Band Roformer inference utilities.

The package exposes:
- :class:`MelBandRoformer` – the neural network architecture
- :func:`get_model_from_config` – helper to instantiate from YAML config files
- :func:`demix_track` – overlap-add inference helper used across worzpro demos
- Console entry points ``melband-roformer-infer`` and ``melband-roformer-download``
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

__version__ = "0.1.0"

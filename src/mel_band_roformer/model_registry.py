"""Model registry for Mel-Band Roformer checkpoints.

The registry is built from python-audio-separator's `models.json` "roformer" list so we
share the same naming and config filenames. Each entry exposes a slug, friendly name,
checkpoint filename and config filename plus a coarse category (vocals, instrumental,
karaoke, denoise, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_PACKAGE_ROOT = Path(__file__).resolve().parent
_MODEL_DATA_PATH = _PACKAGE_ROOT / "data" / "melband_models.json"


@dataclass(frozen=True)
class MelBandModel:
    slug: str
    name: str
    checkpoint: str
    config: str
    category: str

    @property
    def default_sources(self) -> List[str]:
        # All mel-band roformer checkpoints emit 2 stems. Category hints intent.
        if self.category in {"instrumental", "instvoc"}:
            return ["instrumental", "vocals"]
        if self.category in {"karaoke", "vocals", "general"}:
            return ["vocals", "other"]
        if self.category == "denoise":
            return ["clean", "residual"]
        if self.category in {"dereverb", "crowd"}:
            return ["dry", "wet"]
        return ["vocals", "other"]


class ModelRegistry:
    def __init__(self):
        data = json.loads(_MODEL_DATA_PATH.read_text())
        self._models: Dict[str, MelBandModel] = {}
        for slug, meta in data["models"].items():
            model = MelBandModel(
                slug=slug,
                name=meta["name"],
                checkpoint=meta["checkpoint"],
                config=meta["config"],
                category=meta.get("category", "general"),
            )
            self._models[slug] = model
        # simple lookup by normalized name or checkpoint
        self._by_name = {model.name.lower(): model.slug for model in self._models.values()}
        self._by_checkpoint = {model.checkpoint.lower(): model.slug for model in self._models.values()}

    def list(self, category: Optional[str] = None) -> List[MelBandModel]:
        if category is None:
            return sorted(self._models.values(), key=lambda m: m.name)
        return sorted(
            (m for m in self._models.values() if m.category == category.lower()),
            key=lambda m: m.name,
        )

    def categories(self) -> List[str]:
        return sorted({m.category for m in self._models.values()})

    def get(self, key: str) -> MelBandModel:
        normalized = key.lower()
        if normalized in self._models:
            return self._models[normalized]
        if normalized in self._by_name:
            return self._models[self._by_name[normalized]]
        if normalized in self._by_checkpoint:
            return self._models[self._by_checkpoint[normalized]]
        raise KeyError(f"Unknown Mel-Band Roformer model: {key}")

    def search(self, term: str) -> List[MelBandModel]:
        normalized = term.lower()
        return [
            m for m in self._models.values()
            if normalized in m.name.lower() or normalized in m.checkpoint.lower()
        ]

    def as_table(self, category: Optional[str] = None) -> str:
        rows = self.list(category)
        if not rows:
            return "No models registered."
        name_w = max(len(m.name) for m in rows)
        cat_w = max(len(m.category) for m in rows)
        lines = [f"{'Name'.ljust(name_w)}  {'Category'.ljust(cat_w)}  Checkpoint"]
        lines.append("-" * len(lines[0]))
        for model in rows:
            lines.append(
                f"{model.name.ljust(name_w)}  {model.category.ljust(cat_w)}  {model.checkpoint}"
            )
        return "\n".join(lines)


MODEL_REGISTRY = ModelRegistry()

# Default model - MelBand Roformer Kim is the original and recommended model for vocal separation
DEFAULT_MODEL = "melband-roformer-kim-vocals"

__all__ = ["MelBandModel", "ModelRegistry", "MODEL_REGISTRY", "DEFAULT_MODEL"]

# MelBand-RoFormer-Infer

**Production-ready, inference-only toolkit for Mel-Band RoFormer audio source separation**

MelBand-RoFormer-Infer provides a clean, lightweight API for running music source separation inference using Mel-Band RoFormer models with automatic checkpoint management.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/melband-roformer-infer)](https://pypi.org/project/melband-roformer-infer/)

---

## Why This Exists

Mel-Band RoFormer is a strong architecture for music source separation,
introduced by Lu, Wang, Kong, and Hung (2023) alongside their Band-Split
RoPE Transformer (BS-RoFormer) work. The reference implementation,
[lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer), provides
the model architecture only -- no checkpoint management, no CLI, no packaging
for downstream use. Trained checkpoints -- including the widely-used vocal
separation model trained by Kimberley Jensen -- are typically distributed
through [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
(which pulls in the full Ultimate Vocal Remover GUI stack) or through
individual community members' personal Hugging Face accounts -- hosts that
can and do vanish or get renamed without warning (see the jarredou account
deletion and the broader 2026-07 registry audit in `CHANGELOG.md`: 37 of the
89 bulk-imported registry entries are currently fully usable, 36 checkpoints
are dead, and 10 models are fully dead).

MelBand-RoFormer-Infer reprovides the architecture as a clean,
pip-installable, inference-only package: no training code, no GUI
dependency, a versioned model registry that can be repointed at a new host
by editing one JSON file (no code change), and sha256-verified auto-download
so a corrupted or tampered checkpoint is never silently loaded.

---

## Acknowledgments

This project builds upon the excellent work of several open-source projects:

- **[Mel-Band-Roformer-Vocal-Model](https://huggingface.co/KimberleyJSN/melbandroformer)** by Kimberley Jensen -- original model training and the recommended default vocal-separation checkpoint
- **[BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)** by Phil Wang (lucidrains) -- PyTorch implementation of the Mel-Band / Band-Split RoPE Transformer architecture
- **[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)** by Andrew Beveridge (nomadkaraoke) -- the `models.json` registry this package's model list was bulk-imported from, plus pre-trained checkpoints and configurations for many registry entries
- **Original Research** -- Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong, and Yun-Ning Hung for the Band-Split RoPE Transformer paper (see Citation below)

## Citation

If you use MelBand-RoFormer-Infer in your research, please cite the original paper:

```bibtex
@inproceedings{lu2024music,
    title     = {Music Source Separation with Band-Split RoPE Transformer},
    author    = {Lu, Wei-Tsung and Wang, Ju-Chiang and Kong, Qiuqiang and Hung, Yun-Ning},
    booktitle = {ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    pages     = {481--485},
    year      = {2024},
    publisher = {IEEE},
    doi       = {10.1109/ICASSP48485.2024.10446843}
}
```

Also available as a preprint: [arXiv:2309.02612](https://arxiv.org/abs/2309.02612).

---

## Features

- **Inference Only**: Lightweight package focused on production inference
- **Auto-Download**: the default model is fetched on first use and sha256-verified against recorded checksums
- **Model Registry**: 89 catalogued models -- vocals, instrumentals, karaoke, denoise, dereverb, and more (see the availability note below)
- **CLI Tools**: `melband-roformer-infer` and `melband-roformer-download` commands
- **Python API**: Clean programmatic interface

## Scope

**In scope**: inference (forward pass) with the Mel-Band RoFormer
architecture; an 89-model registry (`src/mel_band_roformer/data/melband_models.json`)
spanning vocals, instrumental, karaoke, denoise, dereverb, crowd, general,
and aspiration checkpoints; automatic, manual, and configurable-directory
checkpoint management with sha256 verification; a standalone download CLI.

**Out of scope, forever**:
- Training or fine-tuning code -- this package only ever runs a forward pass.
- The Ultimate Vocal Remover GUI itself, or any GUI.
- Bundling or committing checkpoint bytes to this repository's git history
  (see [What This Project Will NEVER Bundle](#what-this-project-will-never-bundle)).

---

## Install

```bash
# Using pip
pip install melband-roformer-infer

# Using UV (recommended)
uv pip install melband-roformer-infer
```

## Quick Start

```bash
# First run auto-downloads the recommended MelBand Roformer Kim model (~913 MB,
# sha256-verified) into ~/.cache/melband-roformer-infer/ -- no separate download step needed
melband-roformer-infer --input_folder ./songs --store_dir ./outputs
```

Every WAV inside `input_folder` produces `*_vocals.wav` and `*_instrumental.wav` stems. Explicit `--config_path`/`--model_path` arguments still work and skip auto-resolution entirely; `--model <slug>` picks a different registry model to auto-resolve.

---

## Python API

```python
from ml_collections import ConfigDict
import torch
import yaml
from mel_band_roformer import DEFAULT_MODEL, ensure_model_assets, get_model_from_config

# Resolves local copies, or downloads (sha256-verified) on first use
ckpt_path, config_path = ensure_model_assets(DEFAULT_MODEL)

config = ConfigDict(yaml.safe_load(open(config_path)))
model = get_model_from_config("mel_band_roformer", config)
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
```

## Recommended Model

**MelBand Roformer Kim** (`melband-roformer-kim-vocals`) by Kimberley Jensen is the recommended default model for vocal separation. It provides excellent quality and is the foundation for many fine-tuned variants.

```python
from mel_band_roformer import DEFAULT_MODEL
print(DEFAULT_MODEL)  # "melband-roformer-kim-vocals"
```

## Available Models

| Model | Category | Description |
|-------|----------|-------------|
| **`melband-roformer-kim-vocals`** | vocals | **Recommended** - Original MelBand Roformer by Kimberley Jensen |
| `melband-roformer-big-beta6` | vocals | Big Beta 6 by unwa |
| `roformer-model-melband-roformer-vocals-by-becruily` | vocals | Vocals by becruily |
| `roformer-model-melband-roformer-instrumental-by-gabox` | instrumental | Instrumental by Gabox |
| `roformer-model-melband-roformer-karaoke-by-becruily` | karaoke | Karaoke by becruily |
| `melband-roformer-denoise-debleed-gabox` | denoise | Denoise Debleed by Gabox |
| `roformer-model-melband-roformer-de-reverb-by-anvuew` | dereverb | De-Reverb by anvuew |
| ... | ... | See `--list-models` for 89 models |

**Categories**: vocals, instrumental, karaoke, denoise, dereverb, crowd, general, aspiration

> **Note on download availability** (re-audited 2026-07-12): this registry is
> bulk-imported from several third-party contributors' Hugging Face repos, some
> of which get renamed or taken down without notice (see `CHANGELOG.md` for the
> 2026-07 audit and the jarredou account deletion). As of the latest audit,
> **37 of the 89 registry models are fully usable** (checkpoint and config both
> live -- all of these carry recorded sha256 checksums); 36 checkpoints are
> dead, and 10 models are fully dead (both checkpoint and config unreachable).
> Run `python tools/check_weights_liveness.py` (needs network access) to
> re-check which models currently have a live download URL before relying on
> one in a pipeline; `--model`/`--category` downloads will print a clear error
> if a URL 404s rather than failing silently.

## Registry Helpers

```python
from mel_band_roformer import MODEL_REGISTRY

# List all categories
print(MODEL_REGISTRY.categories())

# List models by category
for model in MODEL_REGISTRY.list("vocals"):
    print(model.name, model.checkpoint)

# Search models
results = MODEL_REGISTRY.search("karaoke")
for m in results:
    print(m.slug)

# Pretty-print all models
print(MODEL_REGISTRY.as_table())
```

---

## What This Project Will NEVER Bundle

Model weights are **never bundled or committed to this repository**. Every
checkpoint is downloaded at runtime from its registry-recorded source,
sha256-verified against `src/mel_band_roformer/data/checksums.json`, and
cached locally -- a mismatch deletes the file and retries instead of
silently keeping a corrupt checkpoint.

### Where weights live

Downloads default to `~/.cache/melband-roformer-infer/<model-slug>/`. The
location is configurable, resolved in this order:

1. Explicit argument: `--models_dir` (inference CLI), `--output-dir` (download CLI), or `ensure_model_assets(..., models_dir=...)` (API)
2. The `MELBAND_ROFORMER_MODELS_PATH` environment variable
3. The default `~/.cache/melband-roformer-infer/`

A relative `./models` directory (the pre-0.1.4 default) is still searched as a
read fallback, so existing downloads keep working without re-fetching.

### Auto-download

When `melband-roformer-infer` runs without `--model_path`/`--config_path`, the
requested registry model (default: MelBand Roformer Kim) is looked up in the
directories above and downloaded on first use. Downloads are verified against
the sha256 checksums recorded in `src/mel_band_roformer/data/checksums.json`
(71 assets covering every URL that was live in the 2026-07-12 audit); a
mismatch deletes the file and retries instead of keeping a corrupt checkpoint.
Assets without a recorded hash (only reachable via unaudited fallback URLs)
print a warning and fall back to a basic size check.

### Manual download (offline / air-gapped)

The recommended Kim model needs one file (its config ships inside the package):

| File | URL | sha256 |
|------|-----|--------|
| `MelBandRoformer.ckpt` (913,106,900 bytes) | <https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt> | `87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e` |

Place it at
`~/.cache/melband-roformer-infer/melband-roformer-kim-vocals/MelBandRoformer.ckpt`
(or the equivalent path under your `MELBAND_ROFORMER_MODELS_PATH`), and
inference will pick it up without network access. For any other model, the
download URL is the `overrides.json` entry for its checkpoint (or the TRvlvr
fallback) and the expected sha256 is in `data/checksums.json`.

### Download CLI (manual path)

```bash
# List available models
melband-roformer-download --list-models

# Download the recommended model into the cache dir
melband-roformer-download --model melband-roformer-kim-vocals

# Download by category into a custom directory
melband-roformer-download --category karaoke --output-dir ./models
```

---

## Development

```bash
# Clone repository
git clone https://github.com/openmirlab/melband-roformer-infer.git
cd melband-roformer-infer

# Install with UV
uv sync --extra dev

# Install with pip
pip install -e ".[dev]"
```

```bash
uv run pytest -q       # unit tests (network-marked tests deselected by default)
uv run ruff check .    # lint
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

This project includes code and configurations adapted from:
- **BS-RoFormer** (MIT) - Phil Wang
- **python-audio-separator** (MIT) - Andrew Beveridge
- **Mel-Band-Roformer-Vocal-Model** - Kimberley Jensen

---

## Support

For issues and questions:
- **GitHub Issues**: [github.com/openmirlab/melband-roformer-infer/issues](https://github.com/openmirlab/melband-roformer-infer/issues)

---
## Explicit lifecycle API

For applications that need controlled model lifetime, use `MelBandRoformerSession`:

```python
from mel_band_roformer import MelBandRoformerSession
with MelBandRoformerSession() as session:
    manifest = session.infer("input_folder", store_dir="outputs")
    print(manifest[0]["output_id"], manifest[0]["output_path"])
```

`load()` is idempotent, `infer()` requires a ready session, and `release()` frees
the resident model while allowing a later `load()` to rebuild it. `close()` is
terminal and idempotent (and is called by the context manager). `cache_info()` is
read-only: it resolves the same default or custom checkpoint path that `load()`
would use without creating directories or downloading. Devices accept legacy
automatic selection plus explicit `cpu`, `cuda`, `cuda:N`, and `mps`; unavailable
explicit accelerators raise instead of silently falling back. The packaged
`config/checkpoints.toml` is the runtime source for its declared default model's
URLs and SHA-256 metadata; legacy registry records remain a fallback for other
community model variants. Existing CLI and downloader entry points remain lazy.
The returned manifest is JSON-serializable and records each actual file write with
`input_path`, `track_id`, `output_id`, and `output_path`.

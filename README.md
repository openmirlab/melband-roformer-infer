# MelBand-RoFormer-Infer

**Production-ready, inference-only toolkit for Mel-Band RoFormer audio source separation**

MelBand-RoFormer-Infer provides a clean, lightweight API for running music source separation inference using Mel-Band RoFormer models with automatic checkpoint management.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/melband-roformer-infer)](https://pypi.org/project/melband-roformer-infer/)

---

## Features

- **Inference Only**: Lightweight package focused on production inference
- **Auto-Download**: Automatic checkpoint downloads with integrity verification
- **70+ Pre-trained Models**: Vocals, instrumentals, karaoke, denoise, dereverb, and more
- **CLI Tools**: `melband-roformer-infer` and `melband-roformer-download` commands
- **Python API**: Clean programmatic interface
- **Model Registry**: Easy model discovery with search and category filtering

---

## Quick Start

### Installation

```bash
# Using pip
pip install melband-roformer-infer

# Using UV (recommended)
uv pip install melband-roformer-infer
```

### Download Models

```bash
# List available models
melband-roformer-download --list-models

# Download the recommended model (MelBand Roformer Kim)
melband-roformer-download --model melband-roformer-kim-vocals

# Download by category
melband-roformer-download --category karaoke --output-dir ./models

# Download all models
melband-roformer-download --all --output-dir ./models
```

### CLI Inference

```bash
# Using the recommended MelBand Roformer Kim model
melband-roformer-infer \
  --config_path models/melband-roformer-kim-vocals/config_vocals_mel_band_roformer.yaml \
  --model_path models/melband-roformer-kim-vocals/MelBandRoformer.ckpt \
  --input_folder ./songs \
  --store_dir ./outputs
```

Every WAV inside `input_folder` produces `*_vocals.wav` and `*_instrumental.wav` stems.

### Python API

```python
from pathlib import Path
from ml_collections import ConfigDict
import torch
import yaml
from mel_band_roformer import MODEL_REGISTRY, DEFAULT_MODEL, get_model_from_config

# Use the default recommended model (MelBand Roformer Kim)
entry = MODEL_REGISTRY.get(DEFAULT_MODEL)

# Load config and model
config = ConfigDict(yaml.safe_load(open(f"models/{entry.slug}/{entry.config}")))
model = get_model_from_config("mel_band_roformer", config)
model.load_state_dict(torch.load(f"models/{entry.slug}/{entry.checkpoint}", map_location="cpu"))
```

---

## Recommended Model

**MelBand Roformer Kim** (`melband-roformer-kim-vocals`) by Kimberley Jensen is the recommended default model for vocal separation. It provides excellent quality and is the foundation for many fine-tuned variants.

```python
from mel_band_roformer import DEFAULT_MODEL
print(DEFAULT_MODEL)  # "melband-roformer-kim-vocals"
```

---

## Available Models

| Model | Category | Description |
|-------|----------|-------------|
| **`melband-roformer-kim-vocals`** | vocals | **Recommended** - Original MelBand Roformer by Kimberley Jensen |
| `melband-roformer-big-beta6` | vocals | Big Beta 6 by unwa |
| `roformer-model-melband-roformer-vocals-by-gabox` | vocals | Vocals by Gabox |
| `roformer-model-melband-roformer-instrumental-by-gabox` | instrumental | Instrumental by Gabox |
| `roformer-model-mel-roformer-karaoke-aufr33-viperx` | karaoke | Karaoke by aufr33/viperx |
| `roformer-model-mel-roformer-denoise-aufr33` | denoise | Denoise by aufr33 |
| `roformer-model-melband-roformer-de-reverb-by-anvuew` | dereverb | De-Reverb by anvuew |
| ... | ... | See `--list-models` for 70+ models |

**Categories**: vocals, instrumental, karaoke, denoise, dereverb, crowd, general, aspiration

---

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

## Development Installation

```bash
# Clone repository
git clone https://github.com/openmirlab/melband-roformer-infer.git
cd melband-roformer-infer

# Install with UV
uv sync

# Install with pip
pip install -e ".[dev]"
```

---

## Acknowledgments

This project builds upon the excellent work of several open-source projects:

- **[Mel-Band-Roformer-Vocal-Model](https://huggingface.co/KimberleyJSN/melbandroformer)** by Kimberley Jensen - Original model and training
- **[BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)** by Phil Wang (lucidrains) - PyTorch implementation of the RoFormer architecture
- **[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)** by Andrew Beveridge (nomadkaraoke) - Pre-trained checkpoints and model configurations
- **Original Research** - Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong, and Yun-Ning Hung for the Band-Split RoPE Transformer paper

---

## License

MIT License - see [LICENSE](LICENSE) for details.

This project includes code and configurations adapted from:
- **BS-RoFormer** (MIT) - Phil Wang
- **python-audio-separator** (MIT) - Andrew Beveridge
- **Mel-Band-Roformer-Vocal-Model** - Kimberley Jensen

---

## Citation

If you use MelBand-RoFormer-Infer in your research, please cite the original paper:

```bibtex
@inproceedings{Lu2023MusicSS,
    title   = {Music Source Separation with Band-Split RoPE Transformer},
    author  = {Wei-Tsung Lu and Ju-Chiang Wang and Qiuqiang Kong and Yun-Ning Hung},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:261556702}
}
```

---

## Support

For issues and questions:
- **GitHub Issues**: [github.com/openmirlab/melband-roformer-infer/issues](https://github.com/openmirlab/melband-roformer-infer/issues)

---

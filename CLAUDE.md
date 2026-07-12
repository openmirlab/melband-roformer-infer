# melband-roformer-infer -- CLAUDE.md

## Scope

melband-roformer-infer is an inference-only package wrapping Mel-Band
RoFormer music source separation. It reprovides the
[lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)
`MelBandRoformer` architecture as a pip-installable, PyTorch-based CLI +
Python API with automatic checkpoint management: no training code, no UVR
GUI dependency. Given an input folder of WAV files, it produces
`*_vocals.wav` / `*_instrumental.wav` stems per track (or category-specific
stems for non-vocal models). See README.md for the public API, CLI, and
full model registry.

**In scope**: inference (forward pass) only; an 89-model registry
(`src/mel_band_roformer/data/melband_models.json`, bulk-imported from
python-audio-separator's `models.json` "roformer" list) spanning vocals,
instrumental, karaoke, denoise, dereverb, crowd, general, and aspiration
checkpoints; sha256-verified auto-download with a configurable-dir UX
contract (explicit arg > `$MELBAND_ROFORMER_MODELS_PATH` >
`~/.cache/melband-roformer-infer`, legacy `./models` honored as a read
fallback); manual/offline install path; a download CLI
(`melband-roformer-download`) independent of the inference CLI.

**Out of scope, forever**: training/fine-tuning code, the UVR GUI itself,
hosting or mirroring checkpoint bytes in this repo's git history (weights
are always fetched at runtime -- see README's "What This Project Will NEVER
Bundle").

## Module layout

- `src/mel_band_roformer/mel_band_roformer.py` -- the `MelBandRoformer`
  model architecture (from lucidrains/BS-RoFormer), largely unmodified.
  Splits the STFT into perceptually-spaced mel bands (via librosa's mel
  filter bank) rather than BS-Roformer's linear band split; overlapping
  bands are averaged back onto shared frequency bins post-inference.
- `src/mel_band_roformer/attend.py` -- attention core with automatic
  flash/math/mem-efficient SDPA backend selection.
- `src/mel_band_roformer/model_registry.py` -- `MelBandModel` +
  `MODEL_REGISTRY`, backed by `data/melband_models.json` so new models
  don't need a code change. `MODEL_REGISTRY.get()` accepts slug, friendly
  name, or checkpoint filename.
- `src/mel_band_roformer/download.py` -- checkpoint/config download, sha256
  verification, models-dir resolution, and the `melband-roformer-download`
  CLI. `ensure_model_assets()` is the auto-download entry point
  `inference.py` calls on first use. Resolution precedence per asset:
  `data/overrides.json` entry > packaged local file under `configs/`
  (configs only) > `DEFAULT_CKPT_BASE_URL`/`DEFAULT_CONFIG_BASE_URL`
  construction (the fallback most of the registry's ~68 bulk-imported
  entries still rely on, and many of those default-constructed URLs were
  never actually live -- see CHANGELOG.md).
- `src/mel_band_roformer/inference.py` -- the `melband-roformer-infer` CLI:
  folder-batch separation, chunked overlap-add, weights auto-resolve via
  `download.py`. Loads configs through `SafeLoaderWithTuple` so community
  configs' `!!python/tuple` YAML tags never reach a real object constructor.
- `src/mel_band_roformer/utils.py` -- `demix_track` (chunked windowed
  overlap-add inference), `get_model_from_config` (filters a raw config
  down to `MelBandRoformer`'s actual constructor params and restores the
  tuple-typed ones yaml flattens to lists).
- `src/mel_band_roformer/data/melband_models.json` -- the model registry
  data (89 entries).
- `src/mel_band_roformer/data/overrides.json` -- the live patch point for
  dead URLs: when a host 404s, edit this file first, before touching
  `download.py`.
- `src/mel_band_roformer/data/checksums.json` -- recorded sha256 + size for
  the 71 assets whose download URLs were live in the 2026-07-12 audit (53
  checkpoints + 18 configs); assets without a recorded hash fall back to a
  non-empty-size check and print a warning saying so.
- `tools/check_weights_liveness.py` -- HEADs every registry URL; needs
  network, not run in default CI (see Testing below).

## Weights hosting (org constitution article 4)

All 89 registry models download from third-party hosts at runtime; none are
committed to this repo. This registry was bulk-imported from
python-audio-separator's `models.json`, which shares checkpoint/config
filenames but not necessarily a live download URL for each entry -- a
2026-07-12 audit found most of the ~68 bulk-imported (non-overridden)
entries had never been wired to a real URL. As of that audit: **37 of 89
models are fully usable** (checkpoint and config both live, both with
recorded sha256), 36 checkpoints are dead, and 10 models are fully dead
(both checkpoint and config unreachable) -- unchanged from the prior
0.1.2 audit; two HF-hosted overrides (`inst_gaboxFv8.ckpt`,
`inst_gaboxFv8B.ckpt`) were newly confirmed dead in this pass.

The recommended default (`melband-roformer-kim-vocals`, Kimberley Jensen's
original model) is unaffected -- its checkpoint and config are both live and
sha256-verified. Provenance for the karaoke-by-gabox override moved once
already, discovered by outage rather than announcement: the original
`jarredou` Hugging Face account was deleted (confirmed gone during the
2026-07 audit) and the override was repointed to a Politrees/UVR_resources
mirror.

`data/overrides.json` is the single patch point for a future re-host; it
does not require a code change. See README's "What This Project Will NEVER
Bundle" for the user-facing contract (auto-download, manual path, sha256
verification, cache location).

## Testing

Run with `uv run pytest -q` (installs via `uv sync --extra dev`; see
Development below). Test files:

- `tests/test_model_configs.py` -- every bundled/downloaded config loads via
  `yaml.safe_load()`, the `!!python/tuple`-to-tuple conversion runs, and
  models instantiate without beartype errors.
- `tests/test_download.py` -- packaged-config matching and override-URL
  resolution regressions.
- `tests/test_weights_ux.py` -- sha256 verification wiring (a wrong hash
  must delete the file, not ship it) and models-dir resolution precedence,
  offline (temp files + monkeypatched `requests`).
- `tests/test_weights_liveness.py` -- HEADs every registry URL for real.
  Marked `network` and **deselected by default**
  (`addopts = "-m 'not network'"` in pyproject.toml); CI never needs network
  access. Run explicitly before a release: `pytest -m network
  tests/test_weights_liveness.py -v`, or `python
  tools/check_weights_liveness.py` directly.
- `tests/test_twin_backports.py` -- regressions ported from bs-roformer-infer
  (this project's fork sibling) that had drifted out of sync; see
  CHANGELOG.md's `[0.1.3]` entry.

CI (`.github/workflows/test.yml`) matrixes Python 3.10-3.13, all
`not network`-marked, all locally green as of 2026-07-12 (43 passed, 177
deselected on every version -- verified by running `uv run pytest -q`
directly).

## File-top header convention

Every load-bearing module under `src/mel_band_roformer/` starts with a
header of this shape (as the module docstring): one-line title, then 2-3
sentences on what the file is for and why it's shaped this way (including
known failure modes and the fix, where relevant), then a `Reads:` line
naming what it imports from inside the package. This convention is already
in place across the package (see `download.py`, `inference.py`,
`model_registry.py`, `utils.py`, `attend.py`); keep headers in sync as files
change.

## Development

```bash
uv sync --extra dev      # install package + dev deps (pytest, ruff)
uv run pytest -q         # unit tests (network-marked tests deselected)
uv run ruff check .      # lint
```

`pip install -e ".[dev]"` is the pip-only equivalent (used by
`.github/workflows/publish.yml`'s release-gate test run).

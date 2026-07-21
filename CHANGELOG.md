# Changelog

All notable changes to this project are documented in this file.

## [0.1.5] - 2026-07-12

Hotfix: added explicit `numba>=0.61.0`/`llvmlite>=0.44.0` floors. Without
them, installing this package alongside another (e.g. bs-roformer-infer)
in the same environment could make the resolver backtrack numba all the
way down to a ~2021 release whose llvmlite build-checks the Python version
inside `setup.py` at exec time rather than via package metadata — pip/uv
can't skip it, so the build explodes on Python 3.11+. Found by testing a
real, plain `pip install` of both packages together (not just via the
project's own lockfile) immediately after the 0.1.4 release.

## [0.1.4] - 2026-07-12

Weights-UX campaign (org Weights UX contract, constitution art. 4): real
integrity verification, true auto-download, and a configurable weights folder.
Twin change with bs-roformer-infer 0.1.4.

### Added
- **sha256 verification wired into downloads**: `data/checksums.json` records
  the known-good sha256 + size for all 71 assets whose download URLs were live
  in the 2026-07-12 audit (53 checkpoints + 18 configs). Hashes were obtained
  from Hugging Face LFS pointers (authoritative content-address oids) and
  cross-verified against independently downloaded local copies for
  MelBandRoformer.ckpt
  (`87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e`,
  913,106,900 bytes — two local copies) and big_beta6.ckpt (one local copy).
  `download_file`/`verify_file_integrity` now check the hash after every
  download; a mismatch deletes the file and retries instead of keeping a
  corrupt checkpoint. The previously dead `get_file_hash()` helper is now the
  actual verification path. Assets without a recorded hash fall back to the
  old non-empty check and say so.
- **Auto-download on first use**: `melband-roformer-infer` no longer requires
  `--model_path`/`--config_path` — when omitted, the registry model selected
  via the new `--model` flag (default: MelBand Roformer Kim) is resolved from
  the models directory and downloaded (sha256-verified) if missing, via the
  new public `ensure_model_assets()` API (also exported from the package
  root). Explicit paths still win and skip auto-resolution.
- **Configurable weights folder**: downloads now default to
  `~/.cache/melband-roformer-infer/` instead of the CWD-relative `./models`.
  Overridable via the `MELBAND_ROFORMER_MODELS_PATH` env var, the inference
  CLI's `--models_dir`, the download CLI's `--output-dir`, or the API's
  `models_dir=` argument. A legacy `./models` directory is still searched as
  a read fallback so pre-0.1.4 downloads keep working without re-fetching.
- `tests/test_weights_ux.py`: offline unit tests for the hash-verification
  wiring (including a fake-response download that must reject a wrong sha256
  and delete the file), the models-dir resolution order, and a lock that every
  live HF override checkpoint carries a recorded checksum.
- **`.github/workflows/test.yml`**: push/PR-triggered CI, closing a gap found
  in an org-wide audit — the only prior workflow was `publish.yml`, which
  builds/tests on a single pinned Python (3.10) at release time only, so
  nothing ran between releases. Matrix covers Python 3.10-3.13 (all four
  verified locally: 43 passed, 177 deselected — the `network`-marked tests
  that hit real weight hosts, already excluded by `addopts = "-m 'not
  network'"`). A `build` job (needs `test`) does the wheel-from-sdist smoke
  test: `python -m build`, install the wheel into a clean venv, import
  `mel_band_roformer` and touch `MelBandRoformer`, and confirm the wheel
  bundles no `.ckpt` weight file (constitution art. 7). Twin change with
  bs-roformer-infer's test.yml.

### Changed
- Version is now single-sourced from `src/mel_band_roformer/__about__.py` via
  hatch's dynamic version; `pyproject.toml` and `__init__.py` no longer carry
  duplicate literals.
- README: weights section rewritten to state the auto-download behavior, the
  manual download recipe (exact URL + sha256 + target path), and the folder
  resolution order truthfully. The previous "integrity verification" feature
  claim described code that never ran; it is now accurate. Availability note
  refreshed with the 2026-07-12 re-audit numbers: 37 of 89 models fully
  usable, 36 checkpoints dead, 10 models fully dead (unchanged from the
  0.1.2 audit; two HF-hosted overrides, `inst_gaboxFv8.ckpt` and
  `inst_gaboxFv8B.ckpt`, are also confirmed dead).

### Removed
- Google Colab quickstart notebook (`notebooks/quickstart_colab.ipynb`) and
  the README's Colab badge/section — maintaining a separate notebook
  environment alongside the PyPI package was more upkeep than the audience
  justified.

## [0.1.3] - 2026-07-11

Twin-drift backport campaign: bs-roformer-infer was forked from this project and
later accumulated three real bug fixes that were never ported back. Porting them
here, adapted to melband's config/naming, plus two trivial pre-existing nits.

- **Fix**: `inference.py` now loads YAML configs through `SafeLoaderWithTuple`
  instead of plain `yaml.safe_load`. Some community training configs embed
  `!!python/tuple` tags for `multi_stft_resolutions_window_sizes`, which the plain
  safe loader rejects outright; the custom loader downgrades the tag to a list,
  which `utils.get_model_from_config` already converts back to a tuple.
- **Fix**: `inference.py`'s `run_folder` now reads `target_instrument` via
  `getattr(config.training, "target_instrument", None)` instead of direct attribute
  access, so configs that omit the key entirely (rather than setting it to `null`)
  no longer raise `AttributeError`.
- **Fix**: `inference.py`'s `run_folder` now guards the instrumental-stem write with
  `if instruments:`, so a config that resolves to an empty instruments list no
  longer raises `IndexError` on `instruments[0]`.
- **Fix**: `utils.get_model_from_config` now filters the config's `model` section
  through a `valid_params` allowlist before constructing `MelBandRoformer`, so an
  unknown/future config key is dropped instead of raising `TypeError`.
- **Fix**: `utils.demix_track` now falls back from `config.inference.chunk_size` to
  `config.audio.chunk_size` (and finally a hardcoded default) instead of assuming
  `chunk_size` always lives under `inference`, matching older config schema
  variants.
- **Fix**: `attend.py`'s `Attend` gained an optional `scale` constructor override
  (defaults to `None`, preserving the existing `head_dim ** -0.5` default) plus
  `print_once` diagnostics on GPU backend selection, for parity with the twin.
- **Fix**: `download.py`'s CLI `--help` epilog referenced the nonexistent
  `mel-band-roformer-download` command; corrected to the actual console script name,
  `melband-roformer-download` (see `pyproject.toml`'s `[project.scripts]`).
- **Cleanup**: removed 5 orphaned entries from `download.py`'s
  `STATIC_CHECKPOINT_OVERRIDES` -- checkpoint filenames that don't match any
  registry entry in `data/melband_models.json`, so the override could never be
  reached by `get()`/`search()`. The one entry that is reachable
  (`MelBandRoformer.ckpt`, the `DEFAULT_MODEL`'s checkpoint) was kept.
- **Add**: `tests/test_twin_backports.py` -- targeted regression tests for the
  `inference.py`/`utils.py`/`attend.py` fixes above (yaml tuple tag parsing,
  missing target_instrument, empty instruments guard, valid-params filtering,
  chunk_size fallback, `Attend` scale override).

## [0.1.2] - 2026-07-11

Twin-sync campaign: brought this project up to the same standard its twin
bs-roformer-infer reached in its own 0.1.2 release.

- **Fix**: `packaging` was imported in `attend.py` (to gate flash attention on
  torch>=2.0) but never declared as a dependency in `pyproject.toml` -- the same
  bug the twin bs-roformer-infer fixed via community PR #1. Added
  `packaging>=14.1` to `dependencies`.
- **Fix**: full liveness audit of every checkpoint/config URL the download
  pipeline resolves to (121 unique URLs across 89 registry models) found 92
  dead. One is the exact jarredou-account-deletion pattern the twin hit:
  `overrides.json`'s `config_mel_band_roformer_karaoke.yaml` pointed at a
  `huggingface.co/jarredou/...` repo, and that account is now fully gone
  (confirmed 404 on the account page itself). Repointed 43 dead
  checkpoint/config entries -- including the jarredou-tainted one and its
  paired checkpoint (`KaraokeGabox.ckpt`) -- to `Politrees/UVR_resources`, an
  actively-maintained public mirror of the same UVR/roformer model set,
  cross-referenced by exact/case-insensitive filename match and HEAD-verified
  live before adding. Fully-broken models (both checkpoint and config dead)
  dropped from 32 to 10; the `DEFAULT_MODEL` (`melband-roformer-kim-vocals`)
  was never affected.
- **Add**: `tests/test_download.py` -- locks the `DEFAULT_MODEL`'s
  packaged-config invariant and the jarredou-tainted override's new mirror URL,
  so a future silent revert fails tests instead of shipping quietly.
- **Add**: `tools/check_weights_liveness.py` + `tests/test_weights_liveness.py`
  -- a `pytest.mark.network`-gated liveness check (skipped by default; run with
  `pytest -m network`) that HEADs every registry download URL. Ported from the
  twin bs-roformer-infer.
- **Add**: file-top nav headers (title + rationale + `Reads:` deps) across all
  source modules and the test suite, matching the twin's convention.
- **Add**: publish workflow now gates on the test suite (`needs: [test]`)
  before publishing to PyPI.
- **Docs**: README's example model table swapped two currently-dead example
  slugs for live ones, and added a note pointing at
  `tools/check_weights_liveness.py` for anyone relying on a specific model in
  a pipeline.
- **Known limitation**: ~52 registry URLs are still dead (mostly Gabox
  instrumental/vocal checkpoint variants not carried by the mirror used above,
  plus a handful of others) and could not be matched to a live source with
  reasonable confidence in this pass. These predate this release rather than
  being a new regression -- see the twin-sync campaign report for the full
  per-model breakdown. Tracked as follow-up work, not blocking this release.

## [0.1.1] - 2026-01-13

- See git history prior to this file's creation.
## Unreleased

- Added the additive `MelBandRoformerSession` lifecycle facade and package-owned
  `config/checkpoints.toml` checkpoint metadata.
- Repaired the session lifecycle: `load()` is idempotent, `release()` is
  reloadable, and `close()` is terminal/idempotent (including context-manager
  cleanup). `cache_info()` now shares the loader's path resolver without
  downloading or creating cache directories.
- Device selection now validates explicit `cpu`, `cuda`, `cuda:N`, and `mps`
  requests; unavailable explicit accelerators raise while legacy automatic
  CUDA-or-CPU behavior remains unchanged.
- The default model's downloader now reads URL, filename, size, and SHA-256
  metadata from packaged `config/checkpoints.toml`; non-declared legacy model
  variants retain the existing registry fallback.

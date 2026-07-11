# Changelog

All notable changes to this project are documented in this file.

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

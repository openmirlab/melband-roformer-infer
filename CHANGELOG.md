# Changelog

All notable changes to this project are documented in this file.

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

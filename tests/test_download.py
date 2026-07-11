"""Packaged-config and override-resolution regression tests for the download pipeline.

Ported from the twin bs-roformer-infer's test suite, adapted to melband's own
2026-07 outage: the jarredou HF account behind the karaoke-by-gabox config
(config_mel_band_roformer_karaoke.yaml) was deleted, 404/401-ing that override.
Repointed both the checkpoint (KaraokeGabox.ckpt) and config to the Politrees/
UVR_resources mirror (see overrides.json + CHANGELOG.md). Also guards the
DEFAULT_MODEL's packaged-config invariant: melband-roformer-kim-vocals ships
config_vocals_mel_band_roformer.yaml pre-packaged under configs/, so its config
must never need a network fetch.

Reads: mel_band_roformer.download, mel_band_roformer.model_registry.MODEL_REGISTRY
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mel_band_roformer import download as download_module
from mel_band_roformer.model_registry import MODEL_REGISTRY, DEFAULT_MODEL

KARAOKE_GABOX_SLUG = "melband-roformer-karaoke-gabox"
POLITREES_CKPT_URL = (
    "https://huggingface.co/Politrees/UVR_resources/resolve/main/"
    "models/Roformer/MelBand/model_MelBand-Roformer_Karaoke_by-Gabox.ckpt"
)
POLITREES_CONFIG_URL = (
    "https://huggingface.co/Politrees/UVR_resources/resolve/main/"
    "models/Roformer/MelBand/config_MelBand-Roformer_Karaoke_by-Gabox.yaml"
)


class TestPackagedConfigInvariant:
    """The DEFAULT_MODEL's config must resolve to the bundled file, not a network fetch."""

    def test_default_model_is_kim_vocals(self):
        assert DEFAULT_MODEL == "melband-roformer-kim-vocals"

    def test_packaged_config_file_matches_model_config_name(self):
        model = MODEL_REGISTRY.get(DEFAULT_MODEL)
        packaged = download_module.PACKAGE_ROOT / "configs" / model.config
        assert packaged.exists(), (
            f"Bundled config filename must match model.config ({model.config!r}) "
            f"or _copy_packaged_config will never find it and the config always "
            f"gets network-fetched instead."
        )

    def test_config_url_is_none_when_packaged_copy_exists(self):
        """_config_url returns None (meaning: use the local file) since the names line up."""
        model = MODEL_REGISTRY.get(DEFAULT_MODEL)
        assert download_module._config_url(model) is None

    def test_download_config_uses_packaged_copy_without_network(self, tmp_path, monkeypatch):
        """_download_config must hit the packaged-config path, never download_file."""
        model = MODEL_REGISTRY.get(DEFAULT_MODEL)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError(
                "download_file() was called -- packaged config was not used"
            )

        monkeypatch.setattr(download_module, "download_file", _fail_if_called)

        model_dir = tmp_path / model.slug
        model_dir.mkdir(parents=True)
        ok = download_module._download_config(model, model_dir, force=True)

        assert ok is True
        copied = model_dir / model.config
        assert copied.exists()
        assert copied.read_bytes() == (
            download_module.PACKAGE_ROOT / "configs" / model.config
        ).read_bytes()


class TestJarredouOutageRegression:
    """Guards against a silent revert of overrides.json to the dead jarredou URLs.

    The jarredou HF account was deleted (discovered 2026-06 by the twin
    bs-roformer-infer's outage, confirmed independently gone during melband's
    2026-07 audit: huggingface.co/jarredou now 404s). Melband's karaoke-by-gabox
    config override pointed at a jarredou-owned repo and was repointed to the
    Politrees mirror -- this must not silently drift back.
    """

    def test_checkpoint_override_present_for_karaoke_gabox(self):
        model = MODEL_REGISTRY.get(KARAOKE_GABOX_SLUG)
        assert model.checkpoint in download_module.CHECKPOINT_URL_OVERRIDES, (
            f"No checkpoint override registered for {model.checkpoint!r}; "
            f"it would fall back to the default UVR base URL, which does not host this file."
        )

    def test_checkpoint_override_points_at_politrees_mirror(self):
        model = MODEL_REGISTRY.get(KARAOKE_GABOX_SLUG)
        url = download_module._checkpoint_url(model)
        assert url == POLITREES_CKPT_URL
        assert "jarredou" not in url

    def test_config_override_points_at_politrees_mirror(self):
        overrides = download_module.CONFIG_URL_OVERRIDES
        model = MODEL_REGISTRY.get(KARAOKE_GABOX_SLUG)
        assert model.config in overrides, (
            f"No config override registered for {model.config!r}."
        )
        url = overrides[model.config]
        assert url == POLITREES_CONFIG_URL
        assert "jarredou" not in url

    def test_no_override_url_references_the_dead_jarredou_account(self):
        """Broader guard: no registry override should ever point back at jarredou.

        The account is fully gone (confirmed 404 on the account page itself during
        the 2026-07 audit), so any URL under huggingface.co/jarredou/... is dead.
        """
        checkpoints = download_module.CHECKPOINT_URL_OVERRIDES
        configs = download_module.CONFIG_URL_OVERRIDES
        offenders = [
            (name, url)
            for name, url in {**checkpoints, **configs}.items()
            if "huggingface.co/jarredou/" in url
        ]
        assert not offenders, f"Override(s) still point at the dead jarredou account: {offenders}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

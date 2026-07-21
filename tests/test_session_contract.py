"""Offline lifecycle and checkpoint-resolver contracts for MelBandRoformerSession."""

from pathlib import Path

import pytest

from mel_band_roformer.clean_api import MelBandRoformerSession
from mel_band_roformer.download import resolve_model_asset_paths


class _Model:
    def __init__(self):
        self.cpu_calls = 0

    def load_state_dict(self, state):
        return self

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def cpu(self):
        self.cpu_calls += 1
        return self


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "model.ckpt"
    config = tmp_path / "config.yaml"
    config.write_text("training: {}\n")
    return checkpoint, config


def test_session_lifecycle_reuses_then_reloads_and_closes(monkeypatch, tmp_path):
    import mel_band_roformer.utils as utils
    import torch

    checkpoint, config = _paths(tmp_path)
    created = []
    monkeypatch.setattr(utils, "get_model_from_config", lambda *args: created.append(_Model()) or created[-1])
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    session = MelBandRoformerSession(
        model_name="offline-test-model",
        model_path=checkpoint,
        config_path=config,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="ready"):
        session.infer("input")
    assert session.load() is session
    assert session.load() is session
    assert session.status == "ready"
    assert len(created) == 1
    assert session.release() is session
    assert created[0].cpu_calls == 1
    assert session.status == "released"
    session.load()
    assert len(created) == 2
    assert session.close() is session
    assert session.close() is session
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()
    with pytest.raises(RuntimeError, match="ready"):
        session.infer("input")


def test_context_closes_and_failed_load_is_visible(monkeypatch, tmp_path):
    import mel_band_roformer.utils as utils
    import torch

    checkpoint, config = _paths(tmp_path)
    monkeypatch.setattr(utils, "get_model_from_config", lambda *args: _Model())
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    with MelBandRoformerSession(
        model_name="offline-test-model", model_path=checkpoint, config_path=config
    ) as session:
        assert session.status == "ready"
    assert session.status == "closed"

    monkeypatch.setattr(utils, "get_model_from_config", lambda *args: (_ for _ in ()).throw(ValueError("bad model")))
    failed = MelBandRoformerSession(
        model_name="offline-test-model", model_path=checkpoint, config_path=config
    )
    with pytest.raises(ValueError, match="bad model"):
        failed.load()
    assert failed.status == "failed"


def test_cache_info_uses_read_only_toml_default_and_custom_url(monkeypatch, tmp_path):
    import mel_band_roformer.download as download

    cache_root = tmp_path / "cache"
    monkeypatch.setattr(download, "download_file", lambda *args, **kwargs: pytest.fail("download attempted"))
    session = MelBandRoformerSession(models_dir=cache_root)
    info = session.cache_info()
    expected_checkpoint, expected_config = resolve_model_asset_paths(
        session.model_name, cache_root
    )
    assert info["checkpoint_path"] == str(expected_checkpoint)
    assert info["config_path"] == str(expected_config)
    assert info["cached"] is False
    assert not cache_root.exists()

    url = "https://example.invalid/private/custom.ckpt"
    custom = MelBandRoformerSession(
        models_dir=cache_root,
        checkpoint_url=url,
        checkpoint_sha256="0" * 64,
    ).cache_info()
    assert custom["checkpoint_path"] == str(
        cache_root / "melband-roformer-kim-vocals" / "custom.ckpt"
    )
    assert custom["checkpoint_url"] == url
    assert custom["cached"] is False


def test_toml_metadata_drives_default_download(monkeypatch, tmp_path):
    import mel_band_roformer.download as download

    captured = {}
    monkeypatch.setattr(
        download,
        "download_file",
        lambda url, path, *args, **kwargs: captured.update(url=url, path=path, **kwargs) or True,
    )
    monkeypatch.setattr(download, "_download_config", lambda *args, **kwargs: True)
    checkpoint, _ = download.ensure_model_assets("melband-roformer-kim-vocals", models_dir=tmp_path)
    assert checkpoint.name == "MelBandRoformer.ckpt"
    assert captured["url"] == "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt"
    assert captured["expected_sha256"] == "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e"

    captured.clear()
    custom_url = "https://example.invalid/custom/custom-model.ckpt"
    custom_hash = "f" * 64
    custom_checkpoint, _ = download.ensure_model_assets(
        "melband-roformer-kim-vocals",
        models_dir=tmp_path / "custom",
        checkpoint_url=custom_url,
        checkpoint_sha256=custom_hash,
    )
    assert custom_checkpoint.name == "custom-model.ckpt"
    assert captured["url"] == custom_url
    assert captured["expected_sha256"] == custom_hash

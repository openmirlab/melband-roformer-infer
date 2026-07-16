"""Output-manifest regressions for the clean/session inference APIs.

These tests pin the new contract where run_folder() returns the exact files it
wrote, rather than forcing callers to infer outputs from model defaults or
filename conventions. They stay offline by monkeypatching demix_track() and
using tiny temp WAVs.

Reads: mel_band_roformer.inference, mel_band_roformer.clean_api, soundfile, numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from ml_collections import ConfigDict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mel_band_roformer import inference as inference_module
from mel_band_roformer.clean_api import MelBandRoformerSession


class _NoOpModel:
    def eval(self):
        return self


def _write_short_wav(path: Path, sr: int = 8000, n: int = 800, channels: int = 2) -> None:
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal((n, channels)) * 0.05).astype(np.float32)
    sf.write(path, audio, sr)


def _default_like_config() -> ConfigDict:
    return ConfigDict(
        {
            "training": {
                "instruments": ["vocals", "other"],
                "target_instrument": "vocals",
            },
            "inference": {"chunk_size": 100, "num_overlap": 2},
        }
    )


def _fake_demix_track(config, model, mixture, device, first_chunk_time=None):
    channels, length = mixture.shape
    vocals = np.zeros((channels, length), dtype=np.float32)
    return {"vocals": vocals}, 0.01


class TestRunFolderManifest:
    def test_returns_manifest_for_vocals_and_derived_instrumental(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        store_dir = tmp_path / "out"
        track_path = input_dir / "track.wav"
        _write_short_wav(track_path)

        monkeypatch.setattr(inference_module, "demix_track", _fake_demix_track)

        manifest = inference_module.run_folder(
            _NoOpModel(),
            argparse.Namespace(input_folder=input_dir, store_dir=store_dir),
            _default_like_config(),
            device="cpu",
            verbose=True,
        )

        assert manifest == [
            {
                "input_path": str(track_path),
                "track_id": "track",
                "output_id": "vocals",
                "output_path": str(store_dir / "track_vocals.wav"),
            },
            {
                "input_path": str(track_path),
                "track_id": "track",
                "output_id": "instrumental",
                "output_path": str(store_dir / "track_instrumental.wav"),
            },
        ]

    def test_manifest_covers_every_write_for_multiple_inputs(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        store_dir = tmp_path / "out"
        alpha_path = input_dir / "alpha.wav"
        beta_path = input_dir / "beta.wav"
        _write_short_wav(alpha_path)
        _write_short_wav(beta_path)

        monkeypatch.setattr(inference_module, "demix_track", _fake_demix_track)

        manifest = inference_module.run_folder(
            _NoOpModel(),
            argparse.Namespace(input_folder=input_dir, store_dir=store_dir),
            _default_like_config(),
            device="cpu",
            verbose=True,
        )

        assert [(entry["track_id"], entry["output_id"]) for entry in manifest] == [
            ("alpha", "vocals"),
            ("alpha", "instrumental"),
            ("beta", "vocals"),
            ("beta", "instrumental"),
        ]
        assert [entry["input_path"] for entry in manifest] == [
            str(alpha_path),
            str(alpha_path),
            str(beta_path),
            str(beta_path),
        ]
        assert all(Path(entry["output_path"]).exists() for entry in manifest)


class TestSessionManifest:
    def test_session_infer_returns_run_folder_manifest(self, monkeypatch, tmp_path):
        expected_manifest = [
            {
                "input_path": str(tmp_path / "in" / "song.wav"),
                "track_id": "song",
                "output_id": "vocals",
                "output_path": str(tmp_path / "out" / "song_vocals.wav"),
            }
        ]
        captured = {}

        def fake_run_folder(model, args, config, device, verbose=False):
            captured["model"] = model
            captured["args"] = args
            captured["config"] = config
            captured["device"] = device
            captured["verbose"] = verbose
            return expected_manifest

        monkeypatch.setattr(inference_module, "run_folder", fake_run_folder)

        session = MelBandRoformerSession(
            model=_NoOpModel(),
            config=_default_like_config(),
            device="cpu",
        )

        manifest = session.infer(tmp_path / "in", store_dir=tmp_path / "out", verbose=True)

        assert manifest == expected_manifest
        assert captured["args"].input_folder == tmp_path / "in"
        assert captured["args"].store_dir == tmp_path / "out"
        assert captured["config"] == session._config
        assert captured["device"] == "cpu"
        assert captured["verbose"] is True

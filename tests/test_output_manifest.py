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
import pytest
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


def _fake_demix_from_config(config, model, mixture, device, first_chunk_time=None):
    channels, length = mixture.shape
    output_ids = (
        [str(config.training.target_instrument)]
        if getattr(config.training, "target_instrument", None) is not None
        else [str(instrument) for instrument in config.training.instruments]
    )
    return {
        output_id: np.zeros((channels, length), dtype=np.float32)
        for output_id in output_ids
    }, 0.01


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

    def test_single_non_vocal_target_uses_configured_residual_name(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        store_dir = tmp_path / "out"
        track_path = input_dir / "track.wav"
        _write_short_wav(track_path)

        config = ConfigDict(
            {
                "training": {
                    "instruments": ["Guitar", "Other"],
                    "target_instrument": "Guitar",
                },
                "inference": {"chunk_size": 100, "num_overlap": 2},
            }
        )

        monkeypatch.setattr(inference_module, "demix_track", _fake_demix_from_config)

        manifest = inference_module.run_folder(
            _NoOpModel(),
            argparse.Namespace(input_folder=input_dir, store_dir=store_dir),
            config,
            device="cpu",
            verbose=True,
        )

        assert [(entry["output_id"], Path(entry["output_path"]).name) for entry in manifest] == [
            ("Guitar", "track_Guitar.wav"),
            ("Other", "track_Other.wav"),
        ]

    def test_multi_output_model_does_not_derive_extra_instrumental(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        store_dir = tmp_path / "out"
        track_path = input_dir / "track.wav"
        _write_short_wav(track_path)

        config = ConfigDict(
            {
                "training": {
                    "instruments": ["Vocals", "Instrumental"],
                    "target_instrument": None,
                },
                "inference": {"chunk_size": 100, "num_overlap": 2},
            }
        )

        monkeypatch.setattr(inference_module, "demix_track", _fake_demix_from_config)

        manifest = inference_module.run_folder(
            _NoOpModel(),
            argparse.Namespace(input_folder=input_dir, store_dir=store_dir),
            config,
            device="cpu",
            verbose=True,
        )

        assert [(entry["output_id"], Path(entry["output_path"]).name) for entry in manifest] == [
            ("Vocals", "track_Vocals.wav"),
            ("Instrumental", "track_Instrumental.wav"),
        ]

    def test_flac16_output_format_writes_flac_files_at_pcm16(self, tmp_path, monkeypatch):
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
            output_format="flac16",
        )

        assert [entry["output_path"] for entry in manifest] == [
            str(store_dir / "track_vocals.flac"),
            str(store_dir / "track_instrumental.flac"),
        ]
        for entry in manifest:
            path = Path(entry["output_path"])
            assert path.exists()
            info = sf.info(path)
            assert info.format == "FLAC"
            assert info.subtype == "PCM_16"

    def test_wav_s16_output_format_writes_wav_at_pcm16(self, tmp_path, monkeypatch):
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
            output_format="wav_s16",
        )

        for entry in manifest:
            path = Path(entry["output_path"])
            info = sf.info(path)
            assert info.format == "WAV"
            assert info.subtype == "PCM_16"

    def test_unsupported_output_format_raises(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        store_dir = tmp_path / "out"
        track_path = input_dir / "track.wav"
        _write_short_wav(track_path)

        monkeypatch.setattr(
            inference_module, "demix_track", lambda *a, **k: pytest.fail("must not run")
        )

        with pytest.raises(ValueError, match="unsupported output_format"):
            inference_module.run_folder(
                _NoOpModel(),
                argparse.Namespace(input_folder=input_dir, store_dir=store_dir),
                _default_like_config(),
                device="cpu",
                output_format="mp3_320",
            )


class TestSessionManifest:
    def test_session_infer_returns_folder_run_manifest(self, monkeypatch, tmp_path):
        expected_manifest = [
            {
                "input_path": str(tmp_path / "in" / "song.wav"),
                "track_id": "song",
                "output_id": "vocals",
                "output_path": str(tmp_path / "out" / "song_vocals.wav"),
            }
        ]
        captured = {}

        # session.infer() drives run_folder() directly with its resident model --
        # that is the seam this asserts against.
        def fake_run_folder(
            model, args, config, device, verbose=False, output_format="wav_float32"
        ):
            captured["model"] = model
            captured["args"] = args
            captured["config"] = config
            captured["device"] = device
            captured["verbose"] = verbose
            captured["output_format"] = output_format
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
        assert captured["verbose"] is True
        assert captured["output_format"] == "wav_float32"

    def test_session_infer_forwards_explicit_output_format(self, monkeypatch, tmp_path):
        captured = {}

        # session.infer() drives run_folder() directly with its resident model.
        def fake_run_folder(
            model, args, config, device, verbose=False, output_format="wav_float32"
        ):
            captured["output_format"] = output_format
            return []

        monkeypatch.setattr(inference_module, "run_folder", fake_run_folder)

        session = MelBandRoformerSession(
            model=_NoOpModel(),
            config=_default_like_config(),
            device="cpu",
        )

        session.infer(tmp_path / "in", store_dir=tmp_path / "out", output_format="flac16")

        assert captured["output_format"] == "flac16"

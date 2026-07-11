"""Regression tests for the 2026-07 twin-drift backports from bs-roformer-infer.

bs-roformer-infer was forked from this project and later accumulated real bug fixes
that were never ported back. Each is a defensive addition that only changes behavior
on inputs that previously crashed or mis-parsed -- these tests target exactly those
previously-crashing inputs.

Reads: mel_band_roformer.inference
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from ml_collections import ConfigDict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mel_band_roformer import inference as inference_module
from mel_band_roformer.inference import SafeLoaderWithTuple

TUPLE_TAGGED_YAML = """
model:
  multi_stft_resolutions_window_sizes: !!python/tuple
  - 4096
  - 2048
"""


class TestSafeLoaderWithTuple:
    """Some community configs embed !!python/tuple tags; plain safe_load crashes on them."""

    def test_plain_safe_load_rejects_python_tuple_tag(self):
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(TUPLE_TAGGED_YAML)

    def test_safe_loader_with_tuple_parses_tag_as_list(self):
        data = yaml.load(TUPLE_TAGGED_YAML, Loader=SafeLoaderWithTuple)
        sizes = data["model"]["multi_stft_resolutions_window_sizes"]
        assert sizes == [4096, 2048]
        assert isinstance(sizes, list)  # utils.get_model_from_config converts to tuple


class _NoOpModel:
    def eval(self):
        return self


def _write_short_wav(path, sr=8000, n=800, channels=2):
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal((n, channels)) * 0.05).astype(np.float32)
    sf.write(path, audio, sr)


class TestTargetInstrumentFallback:
    """Community configs sometimes omit target_instrument entirely (not even = null)."""

    def test_missing_target_instrument_key_does_not_raise(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        store_dir = tmp_path / "out"
        _write_short_wav(input_dir / "track.wav")

        # No 'target_instrument' key at all under training -- only instruments.
        config = ConfigDict({"training": {"instruments": ["vocals"]}})

        def fake_demix_track(config, model, mixture, device, first_chunk_time=None):
            channels, length = mixture.shape
            return {"vocals": np.zeros((channels, length), dtype=np.float32)}, None

        monkeypatch.setattr(inference_module, "demix_track", fake_demix_track)

        args = argparse.Namespace(input_folder=str(input_dir), store_dir=str(store_dir))

        # Before the fix: config.training.target_instrument raises AttributeError.
        inference_module.run_folder(_NoOpModel(), args, config, device="cpu", verbose=True)

        assert (store_dir / "track_vocals.wav").exists()
        assert (store_dir / "track_instrumental.wav").exists()


class TestInstrumentsGuard:
    """When instruments resolves empty, the instrumental-stem write must be skipped."""

    def test_empty_instruments_does_not_raise_index_error(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        store_dir = tmp_path / "out"
        _write_short_wav(input_dir / "track.wav")

        config = ConfigDict({
            "training": {"instruments": [], "target_instrument": None},
        })

        def fake_demix_track(config, model, mixture, device, first_chunk_time=None):
            return {}, None

        monkeypatch.setattr(inference_module, "demix_track", fake_demix_track)

        args = argparse.Namespace(input_folder=str(input_dir), store_dir=str(store_dir))

        # Before the fix: res[instruments[0]] raises IndexError on an empty list.
        inference_module.run_folder(_NoOpModel(), args, config, device="cpu", verbose=True)

        assert not (store_dir / "track_instrumental.wav").exists()

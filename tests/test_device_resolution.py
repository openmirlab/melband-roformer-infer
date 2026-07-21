"""Regression tests for the "auto" device sentinel and its consolidation.

_resolve_device() (mel_band_roformer.inference) is the single owner of device
resolution: None/""/"auto" all mean cuda:0-if-available-else-cpu, anything else
(e.g. "cpu", "cuda:0") passes through unchanged. _select_device() (CLI/argparse
layer) and MelBandRoformerSession.load() (clean_api) both delegate to it, so
these tests pin the contract at both call sites rather than re-deriving it.

Reads: mel_band_roformer.inference, mel_band_roformer.clean_api
"""

import argparse
import sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mel_band_roformer import inference as inference_module
from mel_band_roformer.clean_api import MelBandRoformerSession


def _expected_auto_device():
    import torch

    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class TestResolveDevice:
    """Direct coverage of inference._resolve_device, the shared core."""

    def test_none_resolves_to_auto_detected_device(self):
        assert inference_module._resolve_device(None) == _expected_auto_device()

    def test_empty_string_resolves_to_auto_detected_device(self):
        assert inference_module._resolve_device("") == _expected_auto_device()

    def test_auto_literal_resolves_identically_to_none(self):
        assert inference_module._resolve_device("auto") == inference_module._resolve_device(None)

    def test_explicit_cpu_passes_through_unchanged(self):
        import torch

        assert inference_module._resolve_device("cpu") == torch.device("cpu")

    def test_explicit_cuda0_passes_through_unchanged(self):
        import torch

        assert inference_module._resolve_device("cuda:0") == torch.device("cuda:0")

    def test_explicit_cuda_index_is_preserved(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        assert inference_module._resolve_device("cuda:1") == torch.device("cuda:1")

    def test_unavailable_explicit_accelerators_raise(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="CUDA"):
            inference_module._resolve_device("cuda")
        with pytest.raises(RuntimeError, match="MPS"):
            inference_module._resolve_device("mps")

    def test_invalid_cuda_index_raises(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        with pytest.raises(RuntimeError, match="index 1"):
            inference_module._resolve_device("cuda:1")


class TestSelectDeviceFromArgs:
    """_select_device (argparse.Namespace entry point used by the CLI)."""

    def test_unset_device_resolves_to_auto_detected_device(self):
        args = argparse.Namespace(device=None)
        assert inference_module._select_device(args) == _expected_auto_device()

    def test_auto_literal_resolves_to_auto_detected_device(self):
        args = argparse.Namespace(device="auto")
        assert inference_module._select_device(args) == _expected_auto_device()

    def test_explicit_cpu_still_passes_through(self):
        import torch

        args = argparse.Namespace(device="cpu")
        assert inference_module._select_device(args) == torch.device("cpu")

    def test_explicit_cuda_string_still_passes_through(self):
        import torch

        args = argparse.Namespace(device="cuda:0")
        assert inference_module._select_device(args) == torch.device("cuda:0")


class _FakeModel:
    """Minimal stand-in for the real nn.Module the session loads."""

    def load_state_dict(self, state):
        self.state = state
        return self

    def to(self, device):
        self.loaded_to = device
        return self

    def eval(self):
        return self


class TestSessionLoadDeviceResolution:
    """MelBandRoformerSession.load() delegates to _resolve_device via clean_api."""

    def _session(self, tmp_path, monkeypatch, *, device):
        import mel_band_roformer.utils as utils_module

        config_path = tmp_path / "config.yaml"
        config_path.write_text("training: {}\n")
        model_path = tmp_path / "fake_model.ckpt"

        monkeypatch.setattr(utils_module, "get_model_from_config", lambda *a, **k: _FakeModel())
        monkeypatch.setattr("torch.load", lambda *a, **k: {})

        return MelBandRoformerSession(
            model_name="unregistered-test-model-xyz",
            model_path=model_path,
            config_path=config_path,
            device=device,
        )

    def test_unset_device_matches_auto_detected_device(self, tmp_path, monkeypatch):
        session = self._session(tmp_path, monkeypatch, device=None).load()
        assert session.device == _expected_auto_device()

    def test_auto_literal_matches_unset_outcome(self, tmp_path, monkeypatch):
        session = self._session(tmp_path, monkeypatch, device="auto").load()
        assert session.device == _expected_auto_device()

    def test_explicit_cpu_still_passes_through(self, tmp_path, monkeypatch):
        import torch

        session = self._session(tmp_path, monkeypatch, device="cpu").load()
        assert session.device == torch.device("cpu")

    def test_cuda_index_reaches_model(self, tmp_path, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        session = self._session(tmp_path, monkeypatch, device="cuda:1").load()
        assert session.device == torch.device("cuda:1")
        assert session._model.loaded_to == torch.device("cuda:1")

"""Real Kim-vocals regression against pristine upstream and the public session.

The two-second model outputs come from lucidrains/BS-RoFormer at the revision
recorded in the fixture. The nine-second written stems record this package's
existing public inference path before any further changes to that path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from ml_collections import ConfigDict

from mel_band_roformer.checkpoints import checkpoint_metadata
from mel_band_roformer.clean_api import MelBandRoformerSession
from mel_band_roformer.utils import get_model_from_config


FIXTURES = Path(__file__).parent / "fixtures/original_kim_golden"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
MODEL_NAME = "melband-roformer-kim-vocals"


@pytest.fixture(scope="module")
def kim_assets():
    if not torch.cuda.is_available():
        pytest.skip("real Kim-vocals checkpoint requires CUDA")
    if (
        torch.__version__ != MANIFEST["torch"]
        or torch.cuda.get_device_name(0) != MANIFEST["cuda_device"]
        or list(torch.cuda.get_device_capability(0)) != MANIFEST["cuda_capability"]
    ):
        pytest.skip("golden output requires the recorded Torch/CUDA device profile")

    cache = Path(
        os.environ.get("MELBAND_ROFORMER_MODELS_PATH", "~/.cache/melband-roformer-infer")
    ).expanduser()
    model_dir = cache / MODEL_NAME
    checkpoint = model_dir / "MelBandRoformer.ckpt"
    config = model_dir / "config_vocals_mel_band_roformer.yaml"
    if not checkpoint.is_file() or not config.is_file():
        pytest.skip("official Kim-vocals checkpoint and config are not cached")

    artifacts = checkpoint_metadata(MODEL_NAME)["artifacts"]
    for kind, expected in (
        ("checkpoint", MANIFEST["checkpoint_sha256"]),
        ("config", MANIFEST["config_sha256"]),
    ):
        assert next(a["sha256"] for a in artifacts if a["kind"] == kind) == expected
    return checkpoint, config


def test_port_model_matches_pristine_original_complete_outputs(kim_assets):
    checkpoint, config_file = kim_assets
    torch.set_num_threads(1)
    torch.manual_seed(MANIFEST["seed"])
    config = ConfigDict(yaml.safe_load(config_file.read_text()))
    model = get_model_from_config("mel_band_roformer", config).eval().to("cuda:0")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)

    with np.load(FIXTURES / "two_seconds.npz") as golden:
        audio = torch.from_numpy(golden["waveform"]).to("cuda:0")
        assert list(audio.shape) == MANIFEST["waveform_shape"]
        with torch.no_grad():
            fp32 = model(audio).cpu().numpy()
            with torch.autocast("cuda"):
                amp = model(audio).cpu().numpy()
        assert list(fp32.shape) == MANIFEST["output_shape"]
        np.testing.assert_array_equal(fp32, golden["output_fp32"])
        np.testing.assert_array_equal(amp, golden["output_amp"])


def test_public_session_writes_complete_baseline_stems(kim_assets, tmp_path):
    checkpoint, config = kim_assets
    torch.set_num_threads(1)
    torch.manual_seed(MANIFEST["seed"])
    with np.load(FIXTURES / "nine_seconds_public.npz") as golden:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        sf.write(inputs / "synthetic.wav", golden["waveform"],
                 MANIFEST["public_baseline"]["sample_rate"], subtype="FLOAT")
        with MelBandRoformerSession(
            model_path=checkpoint, config_path=config, device="cuda:0", progress=False
        ) as session:
            written = session.infer(inputs, store_dir=tmp_path / "outputs")

        assert {entry["output_id"] for entry in written} == {
            "vocals", "instrumental"
        }
        for entry in written:
            actual, sample_rate = sf.read(entry["output_path"], dtype="float32",
                                          always_2d=True)
            assert sample_rate == MANIFEST["public_baseline"]["sample_rate"]
            assert list(actual.shape) == MANIFEST["public_baseline"]["output_shape"]
            np.testing.assert_array_equal(actual, golden[entry["output_id"]])

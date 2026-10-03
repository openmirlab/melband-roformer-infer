"""Capture Kim-vocals model output from the pristine lucidrains architecture.

Set MELBAND_ORIGINAL_DIR to BS-RoFormer at commit
93a07dda7867d4acd8f5bd49ae3f33e1fcbfd8cf. The original checkout is
read-only; this script imports only its model code, not mel_band_roformer.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml


UPSTREAM_REVISION = "93a07dda7867d4acd8f5bd49ae3f33e1fcbfd8cf"
CHECKPOINT_SHA256 = "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e"
CONFIG_SHA256 = "5e380dfa5d5757ac4c2b7f6ef607b93d5058ecff805e7b05ed730a47b90d103c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The original Kim-vocals golden requires CUDA")
    upstream = Path(os.environ["MELBAND_ORIGINAL_DIR"]).resolve()
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                                   text=True).strip()
    if head != UPSTREAM_REVISION:
        raise ValueError(f"Expected original source {UPSTREAM_REVISION}; got {head}")
    changes = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if changes:
        raise ValueError("Original source has tracked changes")
    source_file = upstream / "bs_roformer/mel_band_roformer.py"
    attention_file = upstream / "bs_roformer/attend.py"
    sys.path.insert(0, str(upstream))
    from bs_roformer import MelBandRoformer

    cache = Path(os.environ.get("MELBAND_ROFORMER_MODELS_PATH",
                                "~/.cache/melband-roformer-infer")).expanduser()
    model_dir = cache / "melband-roformer-kim-vocals"
    checkpoint_file = model_dir / "MelBandRoformer.ckpt"
    config_file = model_dir / "config_vocals_mel_band_roformer.yaml"
    if sha256(checkpoint_file) != CHECKPOINT_SHA256 or sha256(config_file) != CONFIG_SHA256:
        raise ValueError("Kim-vocals artifact SHA-256 mismatch")
    config = yaml.safe_load(config_file.read_text())
    kwargs = dict(config["model"])
    kwargs["multi_stft_resolutions_window_sizes"] = tuple(
        kwargs["multi_stft_resolutions_window_sizes"])
    torch.set_num_threads(1)
    torch.manual_seed(12345)
    model = MelBandRoformer(**kwargs).eval().to("cuda:0")
    state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)

    sample_rate = 44100
    time = np.arange(sample_rate * 2, dtype=np.float64) / sample_rate
    left = 0.2 * np.sin(2 * np.pi * 220 * time) * np.exp(-0.7 * time)
    right = 0.15 * np.sin(2 * np.pi * 330 * time) * np.exp(-0.5 * time)
    waveform = np.stack([left, right]).astype(np.float32)[None]
    audio = torch.from_numpy(waveform).to("cuda:0")
    with torch.no_grad():
        output_fp32 = model(audio).cpu().numpy()
        with torch.autocast("cuda"):
            output_amp = model(audio).cpu().numpy()
    if not np.isfinite(output_fp32).all() or not np.isfinite(output_amp).all():
        raise AssertionError("Original model produced nonfinite output")

    fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures/original_kim_golden"
    fixtures.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fixtures / "two_seconds.npz", waveform=waveform,
                        output_fp32=output_fp32, output_amp=output_amp)
    manifest = {
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_model_sha256": sha256(source_file),
        "upstream_attention_sha256": sha256(attention_file),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "seed": 12345,
        "sample_rate": sample_rate,
        "waveform_shape": list(waveform.shape),
        "output_shape": list(output_fp32.shape),
    }
    (fixtures / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Original model: {waveform.shape} -> {output_fp32.shape}")


if __name__ == "__main__":
    main()

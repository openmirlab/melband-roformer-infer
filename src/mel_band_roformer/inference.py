"""CLI entry point for Mel-Band Roformer inference -- folder-batch separation, chunked overlap-add.

Weights auto-resolve: when --model_path/--config_path are omitted, the requested
registry model (default: the recommended Kim vocals model) is looked up in the
models dir ($MELBAND_ROFORMER_MODELS_PATH > ~/.cache/melband-roformer-infer >
legacy ./models) and downloaded sha256-verified on first use via
download.ensure_model_assets -- explicit paths always win and skip all of that.
run_folder() also returns a JSON-serializable manifest of the files it actually
wrote, so Python callers can consume exact output paths without guessing from
model defaults or filename conventions; the CLI keeps the same behavior by
ignoring that return value.
Training configs sometimes embed `!!python/tuple` YAML tags; SafeLoaderWithTuple
downgrades those to plain lists so `yaml.load` never has to execute an arbitrary
Python-object constructor, and utils.get_model_from_config converts the needed
params back to tuples afterward. Falls back to CPU with a warning when CUDA is
unavailable rather than raising, since inference (unlike training) is still usable,
just slow.

Reads: .utils (demix_track, get_model_from_config), .download (ensure_model_assets),
.model_registry (DEFAULT_MODEL), yaml, ml_collections, torch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, TypedDict

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from ml_collections import ConfigDict
from tqdm import tqdm

from .download import ensure_model_assets
from .model_registry import DEFAULT_MODEL
from .utils import demix_track, get_model_from_config


class SafeLoaderWithTuple(yaml.SafeLoader):
    """YAML loader that treats !!python/tuple as a list (which utils.py converts to tuple)."""
    pass


def _tuple_constructor(loader, node):
    """Convert !!python/tuple to a list; utils.py will convert to tuple later."""
    return loader.construct_sequence(node)


SafeLoaderWithTuple.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_constructor)


def _ensure_wav_inputs(input_folder: Path) -> list[Path]:
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    wav_files = sorted(input_folder.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_folder}")
    return wav_files


def _resolve_output_dir(store_dir: Path) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


def _format_iterable(paths: Iterable[Path], verbose: bool) -> Iterable[Path]:
    return tqdm(paths, desc="Tracks", unit="track") if not verbose else paths


class OutputManifestEntry(TypedDict):
    """JSON-serializable record of one file written by run_folder()."""

    input_path: str
    track_id: str
    output_id: str
    output_path: str


OutputManifest = list[OutputManifestEntry]


def _resolve_output_ids(config: ConfigDict) -> list[str]:
    instruments = [str(instr) for instr in config.training.instruments]
    target_instrument = getattr(config.training, "target_instrument", None)
    if target_instrument is not None:
        return [str(target_instrument)]
    return instruments


def _record_written_output(
    manifest: OutputManifest,
    *,
    input_path: Path,
    output_id: str,
    output_path: Path,
) -> None:
    manifest.append(
        {
            "input_path": str(input_path),
            "track_id": input_path.stem,
            "output_id": output_id,
            "output_path": str(output_path),
        }
    )


def run_folder(model, args, config, device, verbose: bool = False) -> OutputManifest:
    start_time = time.time()
    model.eval()

    input_folder = Path(args.input_folder).expanduser()
    store_dir = _resolve_output_dir(Path(args.store_dir).expanduser())
    all_mixtures_path = _ensure_wav_inputs(input_folder)
    total_tracks = len(all_mixtures_path)
    print(f"Total tracks found: {total_tracks}")

    instruments = _resolve_output_ids(config)

    iterable = _format_iterable(all_mixtures_path, verbose)
    manifest: OutputManifest = []

    first_chunk_time = None

    for track_number, path in enumerate(iterable, 1):
        print(f"\nProcessing track {track_number}/{total_tracks}: {path.name}")

        mix, sr = sf.read(path)
        original_mono = False
        if len(mix.shape) == 1:
            original_mono = True
            mix = np.stack([mix, mix], axis=-1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)

        if first_chunk_time is not None:
            total_length = mixture.shape[1]
            num_chunks = (total_length + config.inference.chunk_size // config.inference.num_overlap - 1) // (config.inference.chunk_size // config.inference.num_overlap)
            estimated_total_time = first_chunk_time * num_chunks
            print(f"Estimated total processing time for this track: {estimated_total_time:.2f} seconds")
            sys.stdout.write(f"Estimated time remaining: {estimated_total_time:.2f} seconds\r")
            sys.stdout.flush()

        res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)

        for instr in instruments:
            vocals_output = res[instr].T
            if original_mono:
                vocals_output = vocals_output[:, 0]

            vocals_path = store_dir / f"{path.stem}_{instr}.wav"
            sf.write(vocals_path, vocals_output, sr, subtype="FLOAT")
            _record_written_output(
                manifest,
                input_path=path,
                output_id=instr,
                output_path=vocals_path,
            )

        if instruments:
            vocals_output = res[instruments[0]].T
            if original_mono:
                vocals_output = vocals_output[:, 0]

            original_mix, _ = sf.read(path)
            instrumental = original_mix - vocals_output

            instrumental_path = store_dir / f"{path.stem}_instrumental.wav"
            sf.write(instrumental_path, instrumental, sr, subtype="FLOAT")
            _record_written_output(
                manifest,
                input_path=path,
                output_id="instrumental",
                output_path=instrumental_path,
            )

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))
    return manifest


def proc_folder(args):
    parser = argparse.ArgumentParser(description="Mel-Band Roformer inference runner")
    parser.add_argument("--model_type", type=str, default="mel_band_roformer")
    parser.add_argument("--config_path", type=Path, default=None,
                        help="path to config yaml file (omit to auto-resolve from --model)")
    parser.add_argument("--model_path", type=Path, default=None,
                        help="path to the .ckpt weights (omit to auto-resolve from --model)")
    parser.add_argument("--model", type=str, default=None,
                        help="registry model slug/name to auto-resolve when paths are omitted "
                             f"(default: {DEFAULT_MODEL})")
    parser.add_argument("--models_dir", type=Path, default=None,
                        help="directory holding downloaded model assets "
                             "(default: $MELBAND_ROFORMER_MODELS_PATH or ~/.cache/melband-roformer-infer; "
                             "a legacy ./models directory is also searched)")
    parser.add_argument("--input_folder", type=Path, required=True, help="folder with songs to process")
    parser.add_argument("--store_dir", type=Path, default=Path("outputs"), help="path to store model outputs")
    parser.add_argument("--device", type=str, default=None, help="torch device string, defaults to auto")
    parser.add_argument("--device_ids", nargs='+', type=int, help='optional list of gpu ids for DataParallel')
    if args is None:
        args = parser.parse_args()
    elif isinstance(args, argparse.Namespace):
        # Already parsed, use as is
        pass
    else:
        args = parser.parse_args(args)

    _resolve_model_assets(args, parser)

    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=SafeLoaderWithTuple))

    model = get_model_from_config(args.model_type, config)
    print(f"Using model weights: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))

    device = _select_device(args)

    if args.device_ids:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for --device_ids usage")
        model = nn.DataParallel(model, device_ids=args.device_ids).to(device)
    else:
        model = model.to(device)

    run_folder(model, args, config, device, verbose=False)


def _resolve_model_assets(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Fill in args.model_path/args.config_path, auto-downloading when both are omitted."""
    model_path = getattr(args, "model_path", None)
    config_path = getattr(args, "config_path", None)
    if (model_path is None) != (config_path is None):
        parser.error("--model_path and --config_path must be given together "
                     "(or both omitted to auto-resolve)")
    if model_path is not None:
        if getattr(args, "model", None):
            parser.error("--model selects a registry model to auto-resolve; "
                         "it cannot be combined with explicit --model_path/--config_path")
        return
    model_key = getattr(args, "model", None) or DEFAULT_MODEL
    args.model_path, args.config_path = ensure_model_assets(
        model_key, models_dir=getattr(args, "models_dir", None)
    )


def _resolve_device(device: str | None) -> torch.device:
    """Resolve None, "", or the literal string "auto" to cuda:0-if-available-else-cpu.

    Any other explicit value (e.g. "cpu", "cuda:0") passes through unchanged.
    """
    if device and device != "auto":
        if device == "cpu":
            return torch.device("cpu")
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    print("CUDA is not available. Falling back to CPU. This will be slow.")
    return torch.device("cpu")


def _select_device(args: argparse.Namespace) -> torch.device:
    return _resolve_device(args.device)


def main():
    """Main entry point for the mel-band-roformer inference script."""
    proc_folder(None)


if __name__ == "__main__":
    main()

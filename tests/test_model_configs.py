"""Test that all model configs load correctly and models can be instantiated.

This test validates:
1. All configs can be loaded with yaml.safe_load() (no !!python/tuple issues)
2. The list-to-tuple conversion in get_model_from_config works
3. Models can be instantiated without beartype errors

Usage:
    pytest tests/test_model_configs.py -v

    # Or run directly:
    python tests/test_model_configs.py
"""

import sys
from pathlib import Path

import pytest
import yaml
from ml_collections import ConfigDict

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mel_band_roformer.utils import get_model_from_config


def find_all_configs() -> list[tuple[str, Path]]:
    """Find all config files in the project."""
    project_root = Path(__file__).parent.parent
    configs = []

    # Bundled configs in src
    bundled_dir = project_root / "src" / "mel_band_roformer" / "configs"
    for cfg in bundled_dir.glob("*.yaml"):
        configs.append((f"bundled:{cfg.name}", cfg))

    # Downloaded model configs in models/
    models_dir = project_root / "models"
    if models_dir.exists():
        for cfg in models_dir.glob("**/*.yaml"):
            configs.append((f"models:{cfg.parent.name}/{cfg.name}", cfg))

    return configs


def load_config_safe(config_path: Path) -> ConfigDict:
    """Load config with yaml.safe_load (as inference.py does)."""
    with open(config_path) as f:
        return ConfigDict(yaml.safe_load(f))


class TestConfigLoading:
    """Test that configs can be loaded with safe_load."""

    @pytest.mark.parametrize("name,config_path", find_all_configs())
    def test_config_loads_with_safe_load(self, name: str, config_path: Path):
        """Config should load without !!python/tuple errors."""
        config = load_config_safe(config_path)
        assert config is not None
        assert "model" in config

    @pytest.mark.parametrize("name,config_path", find_all_configs())
    def test_multi_stft_param_is_list(self, name: str, config_path: Path):
        """After safe_load, multi_stft_resolutions_window_sizes should be a list."""
        config = load_config_safe(config_path)
        param = config.model.get("multi_stft_resolutions_window_sizes")
        if param is not None:
            assert isinstance(param, list), (
                f"Expected list from yaml.safe_load, got {type(param).__name__}"
            )


class TestModelInstantiation:
    """Test that models can be instantiated with the list-to-tuple fix."""

    @pytest.mark.parametrize("name,config_path", find_all_configs())
    def test_model_instantiates(self, name: str, config_path: Path):
        """Model should instantiate without beartype errors."""
        config = load_config_safe(config_path)

        # Skip if missing required inference params (incomplete config)
        if not hasattr(config, "training"):
            pytest.skip(f"Config missing 'training' section: {name}")

        model = get_model_from_config("mel_band_roformer", config)
        assert model is not None

        # Verify the model has the expected attribute
        assert hasattr(model, "multi_stft_resolutions_window_sizes")
        assert isinstance(model.multi_stft_resolutions_window_sizes, tuple), (
            "Model param should be tuple after conversion"
        )


def main():
    """Run tests directly without pytest."""
    print("=" * 60)
    print("Testing all model configs")
    print("=" * 60)

    configs = find_all_configs()
    if not configs:
        print("No configs found!")
        return 1

    print(f"\nFound {len(configs)} config(s):\n")

    passed = 0
    failed = 0
    skipped = 0

    for name, config_path in configs:
        print(f"Testing: {name}")

        # Test 1: Load with safe_load
        try:
            config = load_config_safe(config_path)
            print(f"  ✅ Config loads with safe_load")
        except Exception as e:
            print(f"  ❌ Failed to load: {e}")
            failed += 1
            continue

        # Test 2: Check param type
        param = config.model.get("multi_stft_resolutions_window_sizes")
        if param is not None:
            if isinstance(param, list):
                print(f"  ✅ multi_stft_resolutions_window_sizes is list (will be converted)")
            else:
                print(f"  ⚠️  Unexpected type: {type(param).__name__}")

        # Test 3: Model instantiation
        if not hasattr(config, "training"):
            print(f"  ⏭️  Skipped model test (missing 'training' section)")
            skipped += 1
            continue

        try:
            model = get_model_from_config("mel_band_roformer", config)
            if model is not None:
                print(f"  ✅ Model instantiated successfully")
                passed += 1
            else:
                print(f"  ❌ Model is None")
                failed += 1
        except Exception as e:
            print(f"  ❌ Model instantiation failed: {e}")
            failed += 1

        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

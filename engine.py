"""
engine.py — Modular Voice Conversion Inference Engine
=====================================================
Wraps rvc-python's RVCInference for timbre transfer.
Handles model discovery, weight auto-download, and conversion.
"""

import os
import sys
import glob
import logging
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import requests
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


# ─── Fairseq Compatibility Patch ──────────────────────────────────────────────
# fairseq uses mutable defaults in @dataclass fields (e.g. `common: CommonConfig
# = CommonConfig()`), which Python 3.11+ rejects. Rather than monkey-patching
# dataclasses (fragile due to import ordering), we directly patch the fairseq
# source file on disk before it gets imported.

_fairseq_patched = False


def _patch_fairseq_configs():
    """
    Recursively patch fairseq source code to replace mutable dataclass
    defaults with field(default_factory=...). Idempotent.
    Checks for a marker file to avoid re-scanning on every run.
    """
    global _fairseq_patched
    if _fairseq_patched:
        return

    # Check marker file
    marker_path = BASE_DIR / "assets" / ".fairseq_patched"
    if marker_path.exists():
        logger.info("✓ Fairseq patch marker found — skipping scan.")
        _fairseq_patched = True
        return

    _fairseq_patched = True

    import re
    import site

    # Find all possible fairseq installations (system + venvs)
    search_paths = site.getsitepackages() + [site.getusersitepackages()]
    # Also check the venv-local site-packages if active
    if hasattr(sys, "prefix"):
        venv_sp = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
        search_paths.append(venv_sp)

    logger.info(f"Searching for fairseq in: {search_paths}")

    any_patched = False

    for sp in search_paths:
        fairseq_root = os.path.join(sp, "fairseq")
        if not os.path.isdir(fairseq_root):
            continue
        
        logger.info(f"Scanning fairseq at: {fairseq_root}")

        for root, dirs, files in os.walk(fairseq_root):
            for file in files:
                if not file.endswith(".py"):
                    continue
                
                file_path = os.path.join(root, file)
                
                # Optimisation: skip if file likely doesn't have dataclasses
                # We can do a quick check when reading
                try:
                    with open(file_path, "r", errors="ignore") as f:
                        content = f.read()

                    original_content = content
                    modified = False

                    # Pattern 1: field(default=Mutable()) -> field(default_factory=Mutable)
                    # We match 'default=CapWord()' and replace with 'default_factory=CapWord'
                    # Note: We use [A-Z] following convention that classes are Capitalized.
                    # This avoids patching function calls like default=some_func().
                    p1 = r"default\s*=\s*([A-Z]\w+)\(\)"
                    if re.search(p1, content):
                        content = re.sub(p1, r"default_factory=\1", content)
                        modified = True

                    # Pattern 2: var: Type = Mutable() -> var: Type = field(default_factory=Mutable)
                    # This requires ensuring 'field' is imported.
                    p2 = r"(\w+:\s*[\w\.]+)\s*=\s*([A-Z]\w+)\(\)"
                    
                    # We only apply P2 if we are inside a dataclass definition usually, but that's hard to parse with regex.
                    # We'll assume strict type hinting with assignment to a Class() constructor implies a mutable default.
                    # We exclude 'field()' calls which are handled by P1 or correct usage.
                    
                    matches = list(re.finditer(p2, content))
                    if matches:
                        # Apply replacements
                        content = re.sub(p2, r"\1 = field(default_factory=\2)", content)
                        modified = True
                        
                        # Ensure 'field' is imported
                        if "from dataclasses import" in content:
                            if "field" not in content.split("from dataclasses import")[1].split("\n")[0]:
                                content = content.replace(
                                    "from dataclasses import dataclass",
                                    "from dataclasses import dataclass, field",
                                )
                        elif "from dataclasses import" not in content:
                             content = "from dataclasses import field\n" + content

                    # Pattern 3: torch.load(..., map_location=...) in checkpoint_utils.py
                    # Fix for PyTorch 2.6+ 'weights_only=True' breaking fairseq checkpoints
                    if file == "checkpoint_utils.py":
                        target_load = 'torch.load(f, map_location=torch.device("cpu"))'
                        replacement_load = 'torch.load(f, map_location=torch.device("cpu"), weights_only=False)'
                        
                        if target_load in content and "weights_only" not in content:
                            content = content.replace(target_load, replacement_load)
                            modified = True

                    if modified and content != original_content:
                        with open(file_path, "w") as f:
                            f.write(content)
                        logger.info(f"✓ Patched: {file_path}")
                        any_patched = True

                except Exception as e:
                    logger.debug(f"Skipped {file_path}: {e}")

    # Create marker file if we successfully scanned (even if no patches needed, we checked)
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.touch()
        logger.info(f"✓ Created patch marker: {marker_path}")
    except Exception as e:
        logger.warning(f"Could not create patch marker: {e}")


_hydra_patched = False

def _patch_hydra_configs():
    """
    Directly patch hydra/conf/__init__.py to replace mutable dataclass
    defaults with field(default_factory=...).
    """
    global _hydra_patched
    if _hydra_patched:
        return
    _hydra_patched = True

    import re
    import site

    # Find all possible hydra installations
    search_paths = site.getsitepackages() + [site.getusersitepackages()]
    if hasattr(sys, "prefix"):
        venv_sp = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
        search_paths.append(venv_sp)

    logger.info(f"Searching for hydra in: {search_paths}")

    for sp in search_paths:
        configs_path = os.path.join(sp, "hydra", "conf", "__init__.py")
        if not os.path.isfile(configs_path):
            continue

        try:
            with open(configs_path, "r") as f:
                content = f.read()

            original_content = content

            # Ensure field import matches fairseq logic (if needed)
            if "from dataclasses import" in content:
                if "field" not in content:
                    content = content.replace(
                        "from dataclasses import dataclass",
                        "from dataclasses import dataclass, field",
                    )
            else:
                content = "from dataclasses import field\n" + content

            # Replace patterns: var: Type = Class() -> var: Type = field(default_factory=Class)
            content = re.sub(
                r'(\w+:\s*[\w\.]+)\s*=\s*([A-Z]\w+)\(\)',
                r'\1 = field(default_factory=\2)',
                content,
            )

            if content != original_content:
                with open(configs_path, "w") as f:
                    f.write(content)
                logger.info(f"✓ Patched hydra configs: {configs_path}")
            else:
                logger.info(f"✓ hydra configs already patched or correct: {configs_path}")
        except Exception as e:
            logger.warning(f"Could not patch hydra at {configs_path}: {e}")


_fairseq_init_patched = False

def _patch_fairseq_init():
    """
    Patch fairseq/dataclass/initialize.py to skip MISSING values
    during hydra_init to avoid omegaconf ValidationError.
    """
    global _fairseq_init_patched
    if _fairseq_init_patched:
        return
    _fairseq_init_patched = True

    import site

    # Find all possible fairseq installations
    search_paths = site.getsitepackages() + [site.getusersitepackages()]
    if hasattr(sys, "prefix"):
        venv_sp = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
        search_paths.append(venv_sp)

    logger.info(f"Searching for fairseq init in: {search_paths}")

    for sp in search_paths:
        init_path = os.path.join(sp, "fairseq", "dataclass", "initialize.py")
        if not os.path.isfile(init_path):
            continue

        try:
            with open(init_path, "r") as f:
                content = f.read()

            original_content = content

            # 1. Import MISSING from omegaconf
            if "from omegaconf import DictConfig, OmegaConf" in content and "MISSING" not in content:
                content = content.replace(
                    "from omegaconf import DictConfig, OmegaConf",
                    "from omegaconf import DictConfig, OmegaConf, MISSING"
                )

            # 2. Import dataclasses if not present
            if "import dataclasses" not in content:
                content = content.replace("import logging", "import logging\nimport dataclasses")

            # 3. Skip MISSING values (both omegaconf and dataclasses) in hydra_init loop
            # Regex to match the loop and injection point
            pattern = r"(for k in FairseqConfig\.__dataclass_fields__:\s+v = FairseqConfig\.__dataclass_fields__\[k\]\.default)"
            # Note: We use \s+ which swallows the newline and indentation of the original second line, 
            # so we must be careful to reconstruct it correctly.
            # Actually, let's use a simpler replacement that assumes standard indentation found in the file (4 spaces).
            
            # The file has:
            #     for k in FairseqConfig.__dataclass_fields__:
            #         v = FairseqConfig.__dataclass_fields__[k].default
            
            # We want:
            #     for k in FairseqConfig.__dataclass_fields__:
            #         v = FairseqConfig.__dataclass_fields__[k].default
            #         if v is MISSING or v is dataclasses.MISSING:
            #             continue

            # First, revert any previous patch attempts if they differ
            # (Simplest way is just to look for the known good state or overwrite)
            
            # Since the file might already have the "v is MISSING" check from my previous attempt, 
            # I should handle that.
            
            if "or v is dataclasses.MISSING" not in content:
                if "if v is MISSING:" in content:
                    # Update existing patch
                    content = content.replace("if v is MISSING:", "if v is MISSING or v is dataclasses.MISSING:")
                else:
                    # Apply new patch
                    replacement = r"\1\n        if v is MISSING or v is dataclasses.MISSING:\n            continue"
                    content = re.sub(pattern, replacement, content)

            if content != original_content:
                with open(init_path, "w") as f:
                    f.write(content)
                logger.info(f"✓ Patched fairseq init: {init_path}")
            else:
                logger.info(f"✓ fairseq init already patched or correct: {init_path}")

        except Exception as e:
            logger.warning(f"Could not patch fairseq init at {init_path}: {e}")

# ──────────────────────────────────────────────────────────────────────────────

# ─── Constants ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
VOICES_DIR = BASE_DIR / "voices"
OUTPUT_DIR = BASE_DIR / "output"

# HuggingFace URLs for required RVC backbone weights
REQUIRED_ASSETS = {
    "hubert_base.pt": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "desc": "HuBERT feature extractor",
    },
    "rmvpe.pt": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        "desc": "RMVPE pitch estimator",
    },
}


# ─── Weight Auto-Download ─────────────────────────────────────────────────────

def _get_rvc_assets_dir() -> Path:
    """
    Locate the rvc_python package's assets directory so we can place
    hubert_base.pt and rmvpe.pt where the library expects them.
    Falls back to a local 'assets/' folder next to this file.
    """
    try:
        import rvc_python
        pkg_dir = Path(rvc_python.__file__).resolve().parent
        # rvc-python typically keeps weights alongside its package
        # or in a well-known subdirectory
        candidates = [
            pkg_dir / "assets",
            pkg_dir / "rvc" / "assets",
            pkg_dir,
        ]
        for c in candidates:
            if c.exists():
                return c
        # If none exist yet, create the first one
        candidates[0].mkdir(parents=True, exist_ok=True)
        return candidates[0]
    except Exception:
        # Catch all exceptions (ImportError, fairseq compat issues, etc.)
        fallback = BASE_DIR / "assets"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _download_file(url: str, dest: Path, desc: str, progress_callback=None) -> None:
    """Download a file with streaming and optional progress callback."""
    logger.info(f"Downloading {desc} → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = dest.with_suffix(".download")
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total:
                    progress_callback(downloaded / total, desc)

        shutil.move(str(tmp_path), str(dest))
        logger.info(f"✓ Downloaded {desc} ({dest.stat().st_size / 1e6:.1f} MB)")
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def ensure_assets(progress_callback=None) -> dict:
    """
    Check for required backbone weights (hubert_base.pt, rmvpe.pt).
    Downloads any that are missing from HuggingFace.

    Args:
        progress_callback: Optional fn(progress: float, desc: str) for UI feedback.

    Returns:
        Dict mapping asset name → resolved Path.
    """
    assets_dir = _get_rvc_assets_dir()
    resolved = {}

    for name, info in REQUIRED_ASSETS.items():
        # Search in the assets dir and also common sub-paths
        found = None
        search_dirs = [
            assets_dir,
            assets_dir / "hubert",
            assets_dir / "rmvpe",
            BASE_DIR / "assets",
            BASE_DIR / "assets" / "hubert",
            BASE_DIR / "assets" / "rmvpe",
        ]
        for d in search_dirs:
            candidate = d / name
            if candidate.exists() and candidate.stat().st_size > 1_000_000:
                found = candidate
                break

        if found:
            logger.info(f"✓ Found {name} at {found}")
            resolved[name] = found
        else:
            dest = assets_dir / name
            _download_file(info["url"], dest, info["desc"], progress_callback)
            resolved[name] = dest

    return resolved


# ─── Model Discovery ──────────────────────────────────────────────────────────

def discover_models(voices_dir: Optional[Path] = None) -> list[dict]:
    """
    Scan the voices/ folder for .pth RVC model files.

    Returns:
        List of dicts: [{"name": "MarinaAI", "path": Path(...), "size_mb": 52.5}, ...]
    """
    vdir = voices_dir or VOICES_DIR
    if not vdir.exists():
        vdir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Created empty voices directory: {vdir}")
        return []

    models = []
    for pth_file in sorted(vdir.glob("*.pth")):
        name = pth_file.stem  # e.g. "MarinaAI"
        size_mb = pth_file.stat().st_size / (1024 * 1024)
        models.append({
            "name": name,
            "path": pth_file,
            "size_mb": round(size_mb, 1),
        })

    logger.info(f"Discovered {len(models)} voice model(s) in {vdir}")
    return models


# ─── Audio Utilities ──────────────────────────────────────────────────────────

def get_audio_info(file_path: str) -> dict:
    """
    Extract audio metadata using librosa.

    Returns:
        Dict with keys: duration_s, sample_rate, channels, format
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        if y.ndim == 1:
            channels = 1
            duration = len(y) / sr
        else:
            channels = y.shape[0]
            duration = y.shape[1] / sr

        return {
            "duration_s": round(duration, 2),
            "sample_rate": sr,
            "channels": channels,
            "format": Path(file_path).suffix.lstrip(".").upper(),
        }
    except Exception as e:
        logger.error(f"Failed to read audio info: {e}")
        return {"duration_s": 0, "sample_rate": 0, "channels": 0, "format": "?"}


# ─── Voice Converter ──────────────────────────────────────────────────────────

class VoiceConverter:
    """
    High-level voice conversion engine.

    Usage:
        converter = VoiceConverter()
        converter.load_model("MarinaAI")
        converter.convert("input.wav", "output.wav", pitch=12)
    """

    def __init__(self, device: str = "cpu", voices_dir: Optional[Path] = None):
        self.device = device
        self.voices_dir = voices_dir or VOICES_DIR
        self._rvc = None
        self._loaded_model: Optional[str] = None

    def _ensure_rvc(self):
        """Lazy-init the RVCInference instance, patching fairseq first."""
        if self._rvc is None:
            _patch_fairseq_configs()
            _patch_hydra_configs()
            _patch_fairseq_init()
            from rvc_python.infer import RVCInference
            self._rvc = RVCInference(device=self.device)
        return self._rvc

    def load_model(self, model_name: str) -> None:
        """
        Load an RVC voice model by name (without .pth extension).
        Skips re-loading if the same model is already loaded.
        """
        if self._loaded_model == model_name:
            logger.info(f"Model '{model_name}' already loaded — skipping.")
            return

        model_path = self.voices_dir / f"{model_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Voice model not found: {model_path}\n"
                f"Place a .pth file in: {self.voices_dir}"
            )

        rvc = self._ensure_rvc()
        try:
            # Try loading as v2 first (default)
            rvc.load_model(str(model_path), version="v2")
            logger.info(f"✓ Loaded voice model (v2): {model_name}")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                logger.warning(f"v2 load failed ({e}), retrying as v1...")
                try:
                    rvc.load_model(str(model_path), version="v1")
                    logger.info(f"✓ Loaded voice model (v1): {model_name}")
                except Exception as e_v1:
                    raise RuntimeError(f"Failed to load model {model_name} as v1 or v2: {e_v1}")
            else:
                raise

        self._loaded_model = model_name

    def convert(
        self,
        input_path: str,
        output_path: str,
        pitch: int = 12,
        method: str = "rmvpe",
        index_rate: float = 0.75,
        filter_radius: int = 3,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ) -> str:
        """
        Convert a vocal track using the loaded RVC model.

        Args:
            input_path:     Path to input .wav or .mp3 file.
            output_path:    Path for the converted output file.
            pitch:          Pitch transpose in semitones (default +12 for M→F).
                            Set to 0 for timbre-only conversion with no pitch shift.
            method:         f0 extraction method — 'rmvpe' recommended for vocals.
            index_rate:     Feature search ratio (0.0–1.0). Higher = more of the
                            target voice character.
            filter_radius:  Median filter radius for f0 smoothing.
            rms_mix_rate:   Volume envelope blend (0 = output envelope,
                            1 = input envelope).
            protect:        Protection for voiceless consonants (0.0–0.5).

        Returns:
            Absolute path to the output file.
        """
        if self._loaded_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        rvc = self._ensure_rvc()

        # Apply RVC inference settings
        rvc.set_params(
            f0method=method,
            f0up_key=pitch,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )

        rvc.infer_file(input_path, output_path)

        logger.info(
            f"✓ Conversion complete: {input_path} → {output_path} "
            f"(pitch={pitch:+d}, method={method})"
        )
        return str(Path(output_path).resolve())

    def unload_model(self):
        """
        Unload the current model and release resources (GPU memory).
        """
        if self._rvc:
            del self._rvc
            self._rvc = None
        
        self._loaded_model = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                 # MPS doesn't have an explicit empty_cache like CUDA, but gc helps
                 pass
        except ImportError:
            pass
            
        logger.info("✓ Unloaded model and released resources")


# ─── Convenience ──────────────────────────────────────────────────────────────

def get_default_converter(device: str = "cpu") -> VoiceConverter:
    """Create a VoiceConverter with default settings."""
    ensure_assets()
    return VoiceConverter(device=device)

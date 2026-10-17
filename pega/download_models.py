"""
pega.download_models
====================
Downloads pre-trained model weights from the PEGA GitHub release.

Usage
-----
From the command line::

    pega download-models
    pega download-models --dir /path/to/models
    pega download-models --overwrite

From Python::

    from pega.download_models import download_models
    download_models()
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configure before publishing
# ---------------------------------------------------------------------------

_GITHUB_USER = "omicsoldier"
_GITHUB_REPO = "PEGA.py"
_RELEASE_TAG = "v0.1.0-models"

_MODELS = [
    "convolutional_nn_1.h5",  # AMPnet
    "best_model.pth",          # AMP-CG
    "amPEP.model",             # amPEPpy
    "modlamp_RF.joblib",       # modlAMP RF
    "modlamp_SVM.joblib",      # modlAMP SVM
]

_BASE_URL = (
    f"https://github.com/{_GITHUB_USER}/{_GITHUB_REPO}"
    f"/releases/download/{_RELEASE_TAG}"
)

# A file this small can only be a Git LFS pointer stub (real weights are
# multiple MB), left behind when `git clone` ran without LFS support.
_LFS_POINTER_MAX_SIZE = 200
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _is_lfs_pointer_stub(path: Path) -> bool:
    """Return True if ``path`` is a Git LFS pointer file, not real content."""
    try:
        if path.stat().st_size > _LFS_POINTER_MAX_SIZE:
            return False
        with open(path, "rb") as fh:
            return fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False

# ---------------------------------------------------------------------------


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


def download_models(
    model_dir: str | os.PathLike | None = None,
    overwrite: bool = False,
) -> None:
    """Download all PEGA pre-trained model files from GitHub Releases.

    Parameters
    ----------
    model_dir:
        Destination directory.  Defaults to ``pega/models/`` (or the path
        in the ``PEGA_MODEL_DIR`` environment variable if set).
    overwrite:
        If ``True``, re-download files that already exist on disk.
    """
    if model_dir is None:
        model_dir = os.environ.get("PEGA_MODEL_DIR", _default_model_dir())
    dest = Path(model_dir)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model weights to: {dest}\n")

    for filename in _MODELS:
        url = f"{_BASE_URL}/{filename}"
        target = dest / filename

        if target.exists() and not overwrite:
            if _is_lfs_pointer_stub(target):
                print(f"  {filename} — found a Git LFS pointer stub, not the "
                      "real file. Replacing it.")
            else:
                print(f"  {filename} — already exists, skipping.")
                continue

        print(f"  {filename}")
        _download_file(url, target)

    print("\nAll model weights downloaded successfully.")
    print(f"Location: {dest}")


def _download_file(url: str, dest: Path) -> None:
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with (
                open(dest, "wb") as fh,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"    {dest.name}",
                    leave=False,
                ) as bar,
            ):
                for chunk in r.iter_content(chunk_size=8192):
                    fh.write(chunk)
                    bar.update(len(chunk))
    except requests.HTTPError as exc:
        print(
            f"\n  Warning: could not download {dest.name}: {exc}\n"
            f"  URL: {url}\n"
            "  Make sure the GitHub release exists and the file is attached.",
            file=sys.stderr,
        )

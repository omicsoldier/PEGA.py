"""
pega.base
=========
Abstract base class that every PEGA predictor must implement.

Adding a new predictor
----------------------
1. Create a module in ``pega/predictors/``.
2. Define a class that inherits from ``BasePredictor``.
3. Set the five required class attributes.
4. Implement ``is_available()`` and ``score()``.
5. The registry discovers it automatically.
"""

from __future__ import annotations

import abc
import functools
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import ClassVar

import pandas as pd


class BasePredictor(abc.ABC):
    """Abstract base class for all PEGA AMP predictors.

    Class attributes
    ----------------
    name : str
        Short lowercase identifier used in CLI and output column headers.
        Must be unique (e.g. ``"ampnet"``).
    display_name : str
        Human-readable name shown in ``PEGA list`` and log messages
        (e.g. ``"AMPnet"``).
    predictor_id : int
        Stable integer identifier (1–99).
    description : str
        One-sentence description shown by ``PEGA list``.
    category : {"pip", "r", "conda"}
        How the predictor's external dependency is installed.
    """

    name: ClassVar[str]
    display_name: ClassVar[str]
    predictor_id: ClassVar[int]
    description: ClassVar[str]
    category: ClassVar[str]

    @classmethod
    def score_column(cls) -> str:
        """Return the score column name: ``"{name}_score"``."""
        return f"{cls.name}_score"

    @staticmethod
    def models_dir() -> Path:
        """Return the path to the bundled ``pega/models/`` directory."""
        return Path(__file__).resolve().parent / "models"

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Return ``True`` if all runtime dependencies are satisfied.

        Must never raise. Must be fast (no network access, no large imports).
        """

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score all sequences in a FASTA file.

        Returns
        -------
        pandas.DataFrame
            Two columns: ``["seq_name", "<name>_score"]``.
            Scores are in ``[0, 1]``.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _executable_on_path(name: str) -> bool:
        """Return ``True`` if ``name`` is found on the system PATH."""
        return shutil.which(name) is not None

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _r_package_installed(package: str) -> bool:
        """Return ``True`` if an R package is installed inside pega_env.

        Cached — this spawns an Rscript subprocess, and availability can't
        change mid-process, so repeated calls (e.g. once per chunk in
        ``screen_sequences``) would otherwise re-spawn it needlessly.
        """
        # Use the conda environment's Rscript if available
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        rscript = os.path.join(conda_prefix, "bin", "Rscript") if conda_prefix else ""
        if not rscript or not Path(rscript).exists():
            rscript = shutil.which("Rscript") or ""
        if not rscript:
            return False
        result = subprocess.run(
            [rscript, "-e",
             f'if (!requireNamespace("{package}", quietly=TRUE)) quit(status=1)'],
            capture_output=True,
        )
        return result.returncode == 0

    @staticmethod
    def _run_subprocess_with_progress(
        cmd: list[str],
        desc: str,
        cwd: str | None = None,
        env: dict | None = None,
    ) -> subprocess.CompletedProcess:
        """Run ``cmd`` to completion, showing a cosmetic tqdm progress bar.

        stdout/stderr are redirected to temp files, not ``PIPE``. A pipe
        only reaches EOF once *every* process holding its write end has
        closed it — including descendants forked at the C level (e.g. by
        a BLAS library) that never show up in Python's view of the process
        tree. If such a descendant outlives the visible child, reading the
        pipe (``communicate()``) blocks forever even after that child has
        exited — confirmed in production via a py-spy dump showing a
        thread stuck in ``select()`` inside ``communicate()`` while the
        child it was waiting on was already a defunct zombie. A real file
        has no such "wait for every writer" semantics: ``Popen.wait()``
        only tracks the direct child's own exit status, so this can't
        happen regardless of what that child (or anything it spawned)
        does with its file descriptors.

        An earlier version of this method used
        ``subprocess.run(capture_output=True)`` (i.e. ``PIPE``) on a
        background thread specifically to avoid a *different*, simpler
        deadlock (a naive polling loop that never drains the pipe at
        all — see git history) — that fix was necessary but not
        sufficient; this file-based version replaces it.
        """
        with tempfile.TemporaryDirectory() as io_dir:
            stdout_path = os.path.join(io_dir, "stdout.txt")
            stderr_path = os.path.join(io_dir, "stderr.txt")
            result_holder: dict[str, object] = {}

            def _run() -> None:
                # Hand any failure back to the caller. Without this a
                # missing executable or bad cwd raises inside the thread,
                # which only prints a traceback, and the caller then dies
                # on a bare KeyError that names neither cause.
                try:
                    with open(stdout_path, "wb") as out_f, \
                         open(stderr_path, "wb") as err_f:
                        proc = subprocess.run(
                            cmd, stdout=out_f, stderr=err_f, cwd=cwd, env=env,
                        )
                    result_holder["returncode"] = proc.returncode
                except BaseException as exc:  # noqa: BLE001
                    result_holder["error"] = exc

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()

            from tqdm import tqdm
            with tqdm(total=100, desc=desc, unit="%", leave=False) as pbar:
                while thread.is_alive():
                    thread.join(timeout=0.5)
                    if pbar.n < 80:
                        pbar.update(2)
                pbar.update(100 - pbar.n)

            if "error" in result_holder:
                raise result_holder["error"]

            stdout = Path(stdout_path).read_text(errors="replace")
            stderr = Path(stderr_path).read_text(errors="replace")

        return subprocess.CompletedProcess(
            cmd, result_holder["returncode"], stdout, stderr
        )

    @staticmethod
    def _validate_fasta(fasta_path: str | Path) -> Path:
        """Validate that a FASTA file exists, is non-empty, and is pure ASCII."""
        path = Path(fasta_path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"FASTA file not found: {path}\n"
                "Please verify the path and try again."
            )
        if path.stat().st_size == 0:
            raise ValueError(f"FASTA file is empty: {path}")
        try:
            path.read_text(encoding="ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"FASTA file contains non-ASCII characters: {path}\n"
                f"Detail: {exc}\n"
                "All headers and sequences must use standard ASCII characters."
            ) from exc
        return path

    def __repr__(self) -> str:
        status = "available" if self.is_available() else "unavailable"
        return f"<{self.display_name} id={self.predictor_id} [{status}]>"

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if abc.ABC in cls.__bases__:
            return
        required = ("name", "display_name", "predictor_id", "description", "category")
        missing = [a for a in required if not hasattr(cls, a)]
        if missing:
            raise TypeError(
                f"{cls.__name__} is missing required class attributes: "
                + ", ".join(missing)
            )
        valid = {"pip", "r", "conda"}
        if cls.category not in valid:
            raise TypeError(
                f"{cls.__name__}.category must be one of {valid}, "
                f"got '{cls.category}'."
            )

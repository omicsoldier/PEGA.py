"""
pega.predictors.ampnet
======================
AMPnet predictor — convolutional neural network (TensorFlow).

Pre-trained weights
-------------------
File: ``pega/models/convolutional_nn_1.h5``
Download: ``pega download-models``
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from pega.base import BasePredictor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(_AA_ORDER)}
_MAX_LEN = 198

_KD_SCORES: dict[str, float] = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------


class AMPnetPredictor(BasePredictor):
    """Convolutional neural network AMP predictor (TensorFlow)."""

    name = "ampnet"
    display_name = "AMPnet"
    predictor_id = 1
    description = "Convolutional neural network trained on AMP/non-AMP sequences (TensorFlow)."
    category = "pip"

    _model = None

    @classmethod
    def score_column(cls) -> str:
        return "AMPnet_score"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tensorflow  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def _load_model(cls):
        if cls._model is not None:
            return cls._model

        # Suppress TF and absl warnings right before loading.
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        logging.getLogger("absl").setLevel(logging.ERROR)
        logging.getLogger("tensorflow").setLevel(logging.ERROR)

        import tensorflow as tf

        # PEGA already parallelises across predictors/chunks via --jobs;
        # without this, TF also tries to use every core on the node for its
        # own op-level threading, causing severe oversubscription when many
        # predictor threads run TF concurrently (same failure mode fixed for
        # AMPlify's subprocess — see pega.predictors.amplify).
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass  # TF already initialised elsewhere in this process; skip.

        model_path = cls.models_dir() / "convolutional_nn_1.h5"
        if not model_path.exists():
            raise FileNotFoundError(
                f"AMPnet model not found: {model_path}\n"
                "Run 'pega download-models' to download pre-trained weights."
            )
        cls._model = tf.keras.models.load_model(str(model_path))
        return cls._model

    @staticmethod
    def _one_hot_pad(sequence: str) -> np.ndarray:
        matrix = np.zeros((_MAX_LEN, len(_AA_ORDER)), dtype=np.float32)
        for i, aa in enumerate(sequence[:_MAX_LEN]):
            if aa in _AA_INDEX:
                matrix[i, _AA_INDEX[aa]] = 1.0
        return matrix

    @staticmethod
    def _one_hot_batch(sequences: list[str]) -> np.ndarray:
        """Encode a list of sequences as a single (N, MAX_LEN, 20) array."""
        batch = np.zeros(
            (len(sequences), _MAX_LEN, len(_AA_ORDER)), dtype=np.float32
        )
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq[:_MAX_LEN]):
                if aa in _AA_INDEX:
                    batch[i, j, _AA_INDEX[aa]] = 1.0
        return batch

    def score(
        self,
        fasta_path: str | Path,
        batch_size: int = 512,
    ) -> pd.DataFrame:
        """Score sequences with the AMPnet CNN using batch prediction.

        Parameters
        ----------
        fasta_path : str or Path
            Input FASTA file.
        batch_size : int
            Sequences per inference batch. Default 512.
            Increase for more RAM/VRAM; decrease if OOM errors occur.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "AMPnet_score"]``.
        """
        fasta_path = self._validate_fasta(fasta_path)
        model      = self._load_model()

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        if not records:
            raise ValueError(f"No sequences found in {fasta_path}.")

        names  = [r.id for r in records]
        seqs   = [str(r.seq).upper() for r in records]
        scores: list[float] = []

        with tqdm(total=len(seqs), desc="AMPnet", unit="seq") as pbar:
            for i in range(0, len(seqs), batch_size):
                batch_seqs  = seqs[i : i + batch_size]
                batch_array = self._one_hot_batch(batch_seqs)
                preds       = model.predict(batch_array, verbose=0)
                scores.extend(float(p.ravel()[0]) for p in preds)
                pbar.update(len(batch_seqs))

        return pd.DataFrame({"seq_name": names, "AMPnet_score": scores})

"""
pega.predictors.amp_cg
======================
AMP-CG predictor — ESM-2 fine-tuned for AMP classification (PyTorch).

Architecture and inference logic adapted from the original PEGA implementation.

Pre-trained weights
-------------------
File: ``pega/models/best_model.pth``
Download: ``pega download-models``

Reference
---------
Please cite the original AMP-CG publication when using this predictor.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from pega.base import BasePredictor

warnings.filterwarnings("ignore")

_CHECKPOINT = "facebook/esm2_t6_8M_UR50D"


# ---------------------------------------------------------------------------
# Model definition — matches the original PEGA architecture exactly
# ---------------------------------------------------------------------------


def _build_model(device):
    """Build the AMP-CG model architecture."""
    import torch
    import torch.nn as nn
    from transformers import AutoModelForSequenceClassification

    class AMPCGModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                _CHECKPOINT, num_labels=320
            )
            self.bn1 = nn.BatchNorm1d(256)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(320, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.output_layer = nn.Linear(64, 2)
            self.dropout = nn.Dropout(0.0)

        def forward(self, x):
            with torch.no_grad():
                bert_output = self.bert(
                    input_ids=x["input_ids"].to(device),
                    attention_mask=x["attention_mask"].to(device),
                )
            out = self.dropout(bert_output["logits"])
            out = self.relu(self.bn1(self.fc1(out)))
            out = self.relu(self.bn2(self.fc2(out)))
            out = self.relu(self.bn3(self.fc3(out)))
            out = self.output_layer(out)
            return torch.softmax(out, dim=1)

    return AMPCGModel()


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------


class AMPCGPredictor(BasePredictor):
    """ESM-2 transformer AMP predictor (PyTorch + HuggingFace).

    Requires ``torch`` and ``transformers`` packages and
    ``pega/models/best_model.pth``.
    """

    name = "amp_cg"
    display_name = "AMP_CG"
    predictor_id = 5
    description = "ESM-2 protein language model fine-tuned for AMP classification (PyTorch)."
    category = "pip"

    _model = None
    _tokenizer = None
    _device = None

    @classmethod
    def score_column(cls) -> str:
        return "AMP_CG_score"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch          # noqa: F401
            import transformers   # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def _load(cls):
        """Load tokenizer and model (once per process)."""
        if cls._model is not None:
            return cls._model, cls._tokenizer, cls._device

        import logging
        import torch
        from transformers import AutoTokenizer
        import transformers

        # PEGA already parallelises across predictors/chunks via --jobs;
        # without this, PyTorch also tries to use every core on the node for
        # its own op-level threading, causing severe oversubscription when
        # many predictor threads run inference concurrently (same failure
        # mode fixed for AMPlify's subprocess — see pega.predictors.amplify).
        torch.set_num_threads(1)

        # Suppress HuggingFace Hub and transformers informational output.
        # The UNEXPECTED/MISSING keys in the load report are expected —
        # lm_head keys come from the base ESM-2 and are unused;
        # classifier keys are replaced by our custom head.
        transformers.logging.set_verbosity_error()
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

        model_path = cls.models_dir() / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"AMP-CG model not found: {model_path}\n"
                "Run 'pega download-models' to download pre-trained weights."
            )

        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls._tokenizer = AutoTokenizer.from_pretrained(_CHECKPOINT)

        model = _build_model(cls._device)
        state = torch.load(str(model_path), map_location=cls._device, weights_only=False)
        model.load_state_dict(state, strict=False)
        model.to(cls._device)
        model.eval()
        cls._model = model

        return cls._model, cls._tokenizer, cls._device

    def score(
        self,
        fasta_path: str | Path,
        max_len: int = 198,
        batch_size: int = 64,
    ) -> pd.DataFrame:
        """Score sequences with the AMP-CG model.

        Parameters
        ----------
        fasta_path:
            Path to the input FASTA file.
        max_len:
            Maximum tokenisation length (default 198).
        batch_size:
            Number of sequences per inference batch (default 64).

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "AMP_CG_score"]``.
        """
        import torch

        fasta_path = self._validate_fasta(fasta_path)
        model, tokenizer, device = self._load()

        with tqdm(desc="Loading sequences", unit=" seqs") as pbar:
            records = [(r.id, str(r.seq)) for r in SeqIO.parse(str(fasta_path), "fasta")]
            pbar.update(len(records))

        if not records:
            raise ValueError(f"No sequences found in {fasta_path}.")

        ids, seqs = zip(*records)
        total_batches = (len(seqs) + batch_size - 1) // batch_size
        all_probs: list[float] = []

        with tqdm(total=total_batches, desc="AMP-CG", unit=" batch") as pbar:
            for i in range(0, len(seqs), batch_size):
                batch = list(seqs[i : i + batch_size])
                enc = tokenizer(
                    batch,
                    max_length=max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    preds = model(enc)
                    probs = preds.cpu().numpy()[:, 1].tolist()
                all_probs.extend(probs)
                pbar.update(1)

        return pd.DataFrame({"seq_name": list(ids), "AMP_CG_score": all_probs})

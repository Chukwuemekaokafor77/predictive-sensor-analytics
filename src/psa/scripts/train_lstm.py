from __future__ import annotations

import argparse

import numpy as np
import torch

from psa.models.lstm_reconstruction import LSTMReconstructionAnomalyModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="lstm.pt")
    p.add_argument("--seq-len", type=int, default=40)
    p.add_argument("--n-features", type=int, default=4)
    args = p.parse_args()

    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, size=(2000, args.seq_len, args.n_features)).astype(float)

    m = LSTMReconstructionAnomalyModel()
    m.fit(X)

    if m.model is None:
        raise RuntimeError("model is not initialized")

    torch.save(
        {"state_dict": m.model.state_dict(), "threshold": m.threshold, "n_features": args.n_features, "seq_len": args.seq_len},
        args.out,
    )


if __name__ == "__main__":
    main()

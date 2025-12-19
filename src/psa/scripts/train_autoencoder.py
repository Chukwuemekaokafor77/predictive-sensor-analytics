from __future__ import annotations

import argparse

import numpy as np
import torch

from psa.models.autoencoder import AutoencoderAnomalyModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="autoencoder.pt")
    p.add_argument("--n-features", type=int, default=24)
    args = p.parse_args()

    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, size=(8000, args.n_features)).astype(float)

    m = AutoencoderAnomalyModel()
    m.fit(X)

    if m.model is None:
        raise RuntimeError("model is not initialized")

    torch.save({"state_dict": m.model.state_dict(), "threshold": m.threshold, "n_features": args.n_features}, args.out)


if __name__ == "__main__":
    main()

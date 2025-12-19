from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TabularAutoencoder(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        hidden = max(8, n_features // 2)
        bottleneck = max(4, n_features // 4)

        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderAnomalyModel:
    name = "autoencoder"

    def __init__(self, *, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: TabularAutoencoder | None = None
        self.threshold = 0.0

    def fit(self, X: np.ndarray, *, epochs: int = 25, lr: float = 1e-3, batch_size: int = 64) -> None:
        if X.size == 0:
            return
        n_features = int(X.shape[1])
        self.model = TabularAutoencoder(n_features).to(self.device)

        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(x.shape[0], device=self.device)
            for i in range(0, x.shape[0], batch_size):
                idx = perm[i : i + batch_size]
                xb = x[idx]
                opt.zero_grad(set_to_none=True)
                xr = self.model(xb)
                loss = loss_fn(xr, xb)
                loss.backward()
                opt.step()

        errs = self.score(X)
        self.threshold = float(np.quantile(errs, 0.99)) if errs.size else 0.0

    def score(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0 or self.model is None:
            return np.zeros((0,), dtype=float)
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32, device=self.device)
            xr = self.model(x)
            err = torch.mean((xr - x) ** 2, dim=1)
            return err.detach().cpu().numpy().astype(float)


def save_autoencoder_artifact(
    *,
    out_path: str,
    model: AutoencoderAnomalyModel,
    feature_names: list[str],
) -> None:
    if model.model is None:
        raise RuntimeError("model is not initialized")
    payload = {
        "model_family": "autoencoder",
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "threshold": float(model.threshold),
        "state_dict": model.model.state_dict(),
    }
    torch.save(payload, out_path)


def load_autoencoder_artifact(*, path: str, device: str | None = None) -> tuple[AutoencoderAnomalyModel, list[str]]:
    data = torch.load(path, map_location="cpu")
    feature_names = list(data.get("feature_names") or [])
    n_features = int(data.get("n_features") or len(feature_names) or 0)
    if n_features <= 0:
        raise ValueError("Invalid autoencoder artifact: missing n_features/feature_names")

    m = AutoencoderAnomalyModel(device=device)
    m.model = TabularAutoencoder(n_features).to(m.device)
    m.model.load_state_dict(data["state_dict"])
    m.threshold = float(data.get("threshold") or 0.0)
    return m, feature_names

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out, (h, _) = self.encoder(x)
        rep = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        dec_out, _ = self.decoder(rep)
        return self.out(dec_out)


class LSTMReconstructionAnomalyModel:
    name = "lstm_reconstruction"

    def __init__(self, *, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: LSTMAutoencoder | None = None
        self.threshold = 0.0

    def fit(self, X_seq: np.ndarray, *, epochs: int = 15, lr: float = 1e-3, batch_size: int = 32) -> None:
        if X_seq.size == 0:
            return
        n_features = int(X_seq.shape[2])
        self.model = LSTMAutoencoder(n_features).to(self.device)

        x = torch.tensor(X_seq, dtype=torch.float32, device=self.device)
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

        errs = self.score(X_seq)
        self.threshold = float(np.quantile(errs, 0.99)) if errs.size else 0.0

    def score(self, X_seq: np.ndarray) -> np.ndarray:
        if X_seq.size == 0 or self.model is None:
            return np.zeros((0,), dtype=float)
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(X_seq, dtype=torch.float32, device=self.device)
            xr = self.model(x)
            err = torch.mean((xr - x) ** 2, dim=(1, 2))
            return err.detach().cpu().numpy().astype(float)

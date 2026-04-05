"""
dann.py
-------
Unified DANN for single or multi-dimensional E prediction.

- E_source 1-D → trains one model, predict() returns (N,)
- E_source 2-D → trains G models, predict() returns (N, G)

Set n_jobs > 1 (or -1 for all CPUs) to train dimensions in parallel.
All computation runs on CPU only.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import FeatureExtractor, DomainDiscriminator, Predictor, grad_reverse

# Force CPU — no MPS or CUDA
DEVICE = torch.device("cpu")


# -----------------------------------------------------------------------
# Worker function — trains one DANN for one E dimension
# -----------------------------------------------------------------------

def _train_worker(kwargs: dict) -> tuple[int, dict, dict]:
    """
    Train one DANN for one E dimension.
    Returns (g, F_state_dict, P_state_dict).
    """
    g             = kwargs["g"]
    Z_source      = kwargs["Z_source"]
    E_source_col  = kwargs["E_source_col"]
    Z_target      = kwargs["Z_target"]
    input_dim     = kwargs["input_dim"]
    latent_dim    = kwargs["latent_dim"]
    hidden_dim_F  = kwargs["hidden_dim_F"]
    hidden_dim_DP = kwargs["hidden_dim_DP"]
    warmup_epochs = kwargs["warmup_epochs"]
    epochs        = kwargs["epochs"]
    lr            = kwargs["lr"]
    weight_decay  = kwargs["weight_decay"]
    lambda_domain = kwargs["lambda_domain"]
    clip_grad_norm= kwargs["clip_grad_norm"]
    seed          = kwargs["seed"]
    verbose       = kwargs["verbose"]

    torch.manual_seed(seed)

    Zs = torch.tensor(Z_source,     dtype=torch.float32, device=DEVICE)
    Zt = torch.tensor(Z_target,     dtype=torch.float32, device=DEVICE)
    Es = torch.tensor(E_source_col, dtype=torch.float32, device=DEVICE).view(-1, 1)

    F_net = FeatureExtractor(input_dim, hidden_dim_F, latent_dim).to(DEVICE)
    D_net = DomainDiscriminator(latent_dim, hidden_dim_DP).to(DEVICE)
    P_net = Predictor(latent_dim, hidden_dim_DP, output_dim=1).to(DEVICE)

    # Automatically balance source (label=1) vs target (label=0) in the
    # discriminator. pos_weight = N_target / N_source upweights the minority
    # class; equals 1.0 when balanced, so there is no cost for balanced data.
    pos_weight = torch.tensor(
        [len(Z_target) / len(Z_source)], dtype=torch.float32, device=DEVICE
    )
    crit_dom = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    crit_reg = nn.MSELoss()

    opt_FD = torch.optim.Adam(
        list(F_net.parameters()) + list(D_net.parameters()), lr=lr
    )
    opt_D = torch.optim.Adam(
        D_net.parameters(), lr=lr, weight_decay=weight_decay
    )
    opt_FP = torch.optim.Adam(
        list(F_net.parameters()) + list(P_net.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    # warm-up: pre-train F + D so discriminator starts competent
    for _ in range(warmup_epochs):
        fs = F_net(Zs); ft = F_net(Zt)
        loss_D = (crit_dom(D_net(fs), torch.zeros(fs.size(0), 1, device=DEVICE)) +
                  crit_dom(D_net(ft), torch.ones (ft.size(0), 1, device=DEVICE)))
        opt_FD.zero_grad(); loss_D.backward(); opt_FD.step()

    # adversarial training
    for epoch in range(epochs):
        # D step (detached features)
        with torch.no_grad():
            fsd = F_net(Zs); ftd = F_net(Zt)
        loss_D = (crit_dom(D_net(fsd), torch.zeros(fsd.size(0), 1, device=DEVICE)) +
                  crit_dom(D_net(ftd), torch.ones (ftd.size(0), 1, device=DEVICE)))
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # F + P step
        fs = F_net(Zs); ft = F_net(Zt)
        loss_task = crit_reg(P_net(fs), Es)

        p     = epoch / max(epochs - 1, 1)
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
        loss_dom = (
            crit_dom(D_net(grad_reverse(fs, alpha)), torch.zeros(fs.size(0), 1, device=DEVICE)) +
            crit_dom(D_net(grad_reverse(ft, alpha)), torch.ones (ft.size(0), 1, device=DEVICE))
        )
        loss_FP = loss_task + lambda_domain * loss_dom
        opt_FP.zero_grad(); loss_FP.backward()
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                list(F_net.parameters()) + list(P_net.parameters()),
                max_norm=clip_grad_norm,
            )
        opt_FP.step()

        if verbose and (epoch + 1) % 100 == 0:
            print(
                f"  [dim {g}] epoch {epoch+1}/{epochs}  "
                f"task_mse={loss_task.item():.4f}  "
                f"dom_loss={loss_dom.item():.4f}  "
                f"alpha={alpha:.3f}"
            )

    return g, F_net.state_dict(), P_net.state_dict()


# -----------------------------------------------------------------------
# Public class
# -----------------------------------------------------------------------

class DANN:
    """
    Domain-Adversarial Neural Network for cross-domain E prediction.
    Runs on CPU only.

    Automatically handles single or multi-dimensional E:
      - E_source shape (N_s,)    → trains one model,  predict() returns (N,)
      - E_source shape (N_s, G)  → trains G models,   predict() returns (N, G)

    Parameters
    ----------
    input_dim : int
        Number of columns in Z (must be the same for source and target).
    latent_dim : int
        Dimension of the shared feature representation. Default 64.
    hidden_dim_F : int
        Hidden units in the feature extractor. Default 128.
    hidden_dim_DP : int
        Hidden units in discriminator and predictor. Default 64.
    warmup_epochs : int
        Pre-training epochs for feature extractor + discriminator. Default 50.
    epochs : int
        Adversarial training epochs. Default 800.
    lr : float
        Learning rate. Default 1e-3.
    weight_decay : float
        L2 regularisation for discriminator and predictor. Default 0.008.
    lambda_domain : float
        Weight of the domain-adversarial loss. Default 0.05.
    clip_grad_norm : float | None
        Max gradient norm for F+P step. None to disable. Default 5.0.
    seed_base : int
        Dimension g uses seed = seed_base + g. Default 0.
    n_jobs : int
        Number of parallel threads for multi-dim training.
        1 = sequential (default). -1 = use all CPUs.
        Ignored when E is 1-D.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dim_F: int = 128,
        hidden_dim_DP: int = 64,
        warmup_epochs: int = 50,
        epochs: int = 800,
        lr: float = 1e-3,
        weight_decay: float = 0.008,
        lambda_domain: float = 0.05,
        clip_grad_norm: float | None = 5.0,
        seed_base: int = 0,
        n_jobs: int = 1,
    ):
        self.input_dim      = input_dim
        self.latent_dim     = latent_dim
        self.hidden_dim_F   = hidden_dim_F
        self.hidden_dim_DP  = hidden_dim_DP
        self.warmup_epochs  = warmup_epochs
        self.epochs         = epochs
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.lambda_domain  = lambda_domain
        self.clip_grad_norm = clip_grad_norm
        self.seed_base      = seed_base
        self.n_jobs         = n_jobs

        self._F_states: list[dict] = []
        self._P_states: list[dict] = []
        self._G: int | None = None
        self._multidim: bool = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        Z_source: np.ndarray,
        E_source: np.ndarray,
        Z_target: np.ndarray,
        verbose: bool = False,
    ) -> "DANN":
        """
        Train DANN.

        Parameters
        ----------
        Z_source : (N_s, d)
        E_source : (N_s,) or (N_s, G)
        Z_target : (N_t, d)
        verbose  : print loss every 100 epochs if True

        Returns
        -------
        self
        """
        Z_source, E_source, Z_target = self._validate(Z_source, E_source, Z_target)

        G = E_source.shape[1]
        self._G = G

        jobs = [
            dict(
                g=g,
                Z_source=Z_source,
                E_source_col=E_source[:, g],
                Z_target=Z_target,
                input_dim=self.input_dim,
                latent_dim=self.latent_dim,
                hidden_dim_F=self.hidden_dim_F,
                hidden_dim_DP=self.hidden_dim_DP,
                warmup_epochs=self.warmup_epochs,
                epochs=self.epochs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                lambda_domain=self.lambda_domain,
                clip_grad_norm=self.clip_grad_norm,
                seed=self.seed_base + g,
                verbose=verbose,
            )
            for g in range(G)
        ]

        n_workers = self._resolve_n_jobs(G)

        self._F_states = [None] * G
        self._P_states = [None] * G

        if n_workers == 1:
            # sequential
            for job in jobs:
                g, fs, ps = _train_worker(job)
                self._F_states[g] = fs
                self._P_states[g] = ps
                if verbose:
                    print(f"[DANN] dim {g+1}/{G} done")
        else:
            # parallel via threads — safe on all platforms, PyTorch releases GIL
            if verbose:
                print(f"[DANN] training {G} dims with {n_workers} threads")
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_train_worker, job): job["g"] for job in jobs}
                for future in as_completed(futures):
                    g, fs, ps = future.result()
                    self._F_states[g] = fs
                    self._P_states[g] = ps
                    if verbose:
                        print(f"[DANN] dim {g+1}/{G} done")

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """
        Predict E for any domain.

        Parameters
        ----------
        Z : (N, d)

        Returns
        -------
        E_hat : (N,)    if E_source was 1-D
                (N, G)  if E_source was 2-D
        """
        if not self._F_states:
            raise RuntimeError("Call fit() before predict().")
        Z = self._validate_Z(Z)
        N = Z.shape[0]
        Zt = torch.tensor(Z, dtype=torch.float32, device=DEVICE)

        E_hat = np.zeros((N, self._G), dtype=np.float64)
        for g in range(self._G):
            F_net = FeatureExtractor(self.input_dim, self.hidden_dim_F, self.latent_dim).to(DEVICE)
            P_net = Predictor(self.latent_dim, self.hidden_dim_DP, output_dim=1).to(DEVICE)
            F_net.load_state_dict(self._F_states[g])
            P_net.load_state_dict(self._P_states[g])
            F_net.eval(); P_net.eval()
            with torch.no_grad():
                E_hat[:, g] = P_net(F_net(Zt)).view(-1).numpy()

        if not self._multidim:
            return E_hat[:, 0]   # (N,)
        return E_hat             # (N, G)

    def mse(self, Z: np.ndarray, E_true: np.ndarray) -> float | np.ndarray:
        """
        MSE between predictions and true E.

        Returns
        -------
        float    if E_source was 1-D
        (G,)     per-dimension MSE if E_source was 2-D
        """
        E_hat  = self.predict(Z)
        E_true = np.asarray(E_true)
        if not self._multidim:
            return float(np.mean((E_hat - E_true.ravel()) ** 2))
        return np.mean((E_hat - E_true) ** 2, axis=0)  # (G,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_n_jobs(self, G: int) -> int:
        if self.n_jobs == 1:
            return 1
        n_cpu = os.cpu_count() or 1
        if self.n_jobs == -1:
            return min(G, n_cpu)
        return min(self.n_jobs, G, n_cpu)

    def _validate(self, Z_source, E_source, Z_target):
        Z_source = np.asarray(Z_source, dtype=np.float32)
        Z_target = np.asarray(Z_target, dtype=np.float32)
        E_source = np.asarray(E_source, dtype=np.float32)

        if Z_source.ndim != 2:
            raise ValueError(f"Z_source must be 2-D, got {Z_source.shape}")
        if Z_target.ndim != 2:
            raise ValueError(f"Z_target must be 2-D, got {Z_target.shape}")
        if Z_source.shape[1] != Z_target.shape[1]:
            raise ValueError(
                f"Z_source and Z_target must have the same number of features: "
                f"Z_source={Z_source.shape[1]}, Z_target={Z_target.shape[1]}"
            )
        if Z_source.shape[1] != self.input_dim:
            raise ValueError(
                f"Z has {Z_source.shape[1]} features but DANN was built with "
                f"input_dim={self.input_dim}"
            )

        if E_source.ndim == 1:
            E_source = E_source[:, None]
            self._multidim = False
        elif E_source.ndim == 2:
            self._multidim = True
        else:
            raise ValueError(f"E_source must be 1-D or 2-D, got {E_source.shape}")

        if E_source.shape[0] != Z_source.shape[0]:
            raise ValueError(
                f"E_source has {E_source.shape[0]} rows but Z_source has {Z_source.shape[0]}"
            )

        return Z_source, E_source, Z_target

    def _validate_Z(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        if Z.ndim != 2:
            raise ValueError(f"Z must be 2-D, got {Z.shape}")
        if Z.shape[1] != self.input_dim:
            raise ValueError(
                f"Z has {Z.shape[1]} features but model expects {self.input_dim}"
            )
        return Z

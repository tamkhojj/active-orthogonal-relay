"""Refusal recovery operator: the numerical core, model-agnostic.

Implements the mechanistic quantities behind the central claim that refusal is a
self-repairing low-rank code: ablating one direction is repaired downstream by a
rank-k* set of generator directions, and the dimension that must be ablated to
disable refusal is the rank of the measured recovery operator R, k_eff = rank(R).

Everything here is pure tensor math (no model required), so it is unit-testable on
synthetic data; the experiment scripts feed it activations captured by hooks. All
functions are dtype- and device-safe (compute in float32, return on the input
device)."""
from __future__ import annotations

from typing import Dict, Optional
import torch


# --------------------------------------------------------------------------- #
# subspace utilities
# --------------------------------------------------------------------------- #
def orthonormalize(M: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal basis (columns) spanning the columns of M [d, k].

    Uses QR; drops numerically-null columns so the result has full column rank.
    """
    if M.ndim != 2:
        raise ValueError(f"orthonormalize expects [d,k], got {tuple(M.shape)}")
    d, k = M.shape
    Mf = M.float()
    Q, R = torch.linalg.qr(Mf)                 # Q:[d, min(d,k)]
    # keep columns whose diagonal of R is non-negligible (full-rank basis)
    diag = torch.abs(torch.diagonal(R))
    tol = diag.max() * max(d, k) * torch.finfo(Mf.dtype).eps if diag.numel() else 0.0
    keep = diag > tol
    Q = Q[:, : keep.numel()][:, keep]
    return Q.to(M.dtype)


def subspace_projector(U: torch.Tensor) -> torch.Tensor:
    """P = U U^T for orthonormal columns U [d,k]  -> [d,d]."""
    Uf = U.float()
    return (Uf @ Uf.T).to(U.dtype)


def project_coeffs(H: torch.Tensor, U: torch.Tensor):
    """Project rows of H [N,d] onto orthonormal basis U [d,k].

    Returns (coeffs [N,k], reconstruction [N,d]). Device-robust: U is moved to
    H's device so captured (CPU) activations and bases never mismatch.
    """
    Hf = H.float()
    Uf = U.float().to(Hf.device)
    coeffs = Hf @ Uf                           # [N,k]
    recon = coeffs @ Uf.T                       # [N,d]
    return coeffs.to(H.dtype), recon.to(H.dtype)


def scalar_projection(H: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Signed projection of each row of H [N,d] onto unit vector r [d] -> [N]."""
    rf = r.float().to(H.device)
    rf = rf / (rf.norm() + 1e-12)
    return (H.float() @ rf).to(H.dtype)


# --------------------------------------------------------------------------- #
# the recovery operator
# --------------------------------------------------------------------------- #
def fit_ridge_operator(X: torch.Tensor, Y: torch.Tensor, ridge: float = 1e-2) -> torch.Tensor:
    """Least-squares operator R s.t.  y ≈ R x   (columns), i.e. Y ≈ X R^T.

    X : [N, d_in]   damaged-state features (post rank-1 ablation)
    Y : [N, d_out]  regenerated refusal-subspace component downstream
    ridge : relative Tikhonov strength (scaled by tr(X^T X)/d_in)

    Returns R : [d_out, d_in].
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D [N,d]")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"row mismatch: X {tuple(X.shape)} vs Y {tuple(Y.shape)}")
    Xf = X.float()
    Yf = Y.float().to(Xf.device)
    d_in = Xf.shape[1]
    XtX = Xf.T @ Xf                                        # [d_in, d_in]
    lam = ridge * (torch.trace(XtX) / d_in)
    A = XtX + lam * torch.eye(d_in, device=Xf.device, dtype=Xf.dtype)
    W = torch.linalg.solve(A, Xf.T @ Yf)                  # [d_in, d_out]
    return W.T.to(X.dtype)                                 # R = W^T : [d_out, d_in]


def singular_spectrum(M: torch.Tensor) -> torch.Tensor:
    """Singular values of M (descending), float32."""
    return torch.linalg.svdvals(M.float())


def effective_rank(M: torch.Tensor, energy: float = 0.90) -> Dict[str, float]:
    """Several complementary 'effective rank' estimates of an operator M.

    - stable_rank        : ||M||_F^2 / ||M||_2^2   (continuous, robust)
    - participation_ratio: (Σ λ_i)^2 / Σ λ_i^2 ,  λ_i = s_i^2   (spread of energy)
    - energy_rank        : smallest r with cumulative energy >= `energy`
    - spectrum           : the singular values (for plotting)
    """
    s = singular_spectrum(M)
    s = s[s > 0]
    if s.numel() == 0:
        return {"stable_rank": 0.0, "participation_ratio": 0.0,
                "energy_rank": 0, "spectrum": s}
    lam = s ** 2
    stable = (lam.sum() / lam.max()).item()
    pr = ((lam.sum() ** 2) / (lam ** 2).sum()).item()
    cum = torch.cumsum(lam, 0) / lam.sum()
    energy_rank = int((cum < energy).sum().item()) + 1
    return {"stable_rank": stable, "participation_ratio": pr,
            "energy_rank": energy_rank, "spectrum": s}


# --------------------------------------------------------------------------- #
# what gets regenerated, and does it match AOR's generators?
# --------------------------------------------------------------------------- #
def top_subspace(Y: torch.Tensor, k: int, center: bool = True) -> torch.Tensor:
    """Principal subspace (top-k right singular vectors) of rows of Y [N,d].

    Returns orthonormal basis [d, k] spanning the main directions of variation
    of the regenerated signal.
    """
    Yf = Y.float()
    if center:
        Yf = Yf - Yf.mean(0, keepdim=True)
    k = int(max(1, min(k, min(Yf.shape))))
    # economy SVD: Y = U S Vh ; rows of Vh are right singular vectors
    _, _, Vh = torch.linalg.svd(Yf, full_matrices=False)
    return Vh[:k].T.to(Y.dtype)                            # [d,k]


def principal_cosines(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Cosines of principal angles between subspaces spanned by orthonormal U,V."""
    U = orthonormalize(U)
    V = orthonormalize(V)
    c = torch.linalg.svdvals((U.float().T @ V.float()))
    return c.clamp(0.0, 1.0)


def subspace_overlap(U: torch.Tensor, V: torch.Tensor) -> float:
    """Grassmann overlap: mean squared principal cosine in [0,1].

    1.0 = subspaces coincide (up to the smaller dimension); 0.0 = orthogonal.
    """
    c = principal_cosines(U, V)
    if c.numel() == 0:
        return 0.0
    return (c ** 2).mean().item()


# --------------------------------------------------------------------------- #
# recovery curve helpers
# --------------------------------------------------------------------------- #
def recovery_fraction(p_base: torch.Tensor, p_abl: torch.Tensor,
                      eps: float = 1e-8) -> torch.Tensor:
    """rho = p_abl / p_base, elementwise, guarding tiny denominators."""
    pb = p_base.float()
    pa = p_abl.float()
    denom = torch.where(pb.abs() < eps, torch.full_like(pb, eps), pb)
    return pa / denom


def crossing_layer(rho_by_layer: Dict[int, float], thresh: float = 0.5) -> Optional[int]:
    """First layer (ascending) at which recovery fraction rho crosses `thresh`."""
    for layer in sorted(rho_by_layer):
        if rho_by_layer[layer] >= thresh:
            return layer
    return None


def knee_by_energy(curve: torch.Tensor, frac: float = 0.90) -> int:
    """Index (1-based count) where a monotone-ish gain curve reaches `frac` of its
    final value -- used to locate the 'k at which ASR / repair-suppression saturates'.
    """
    c = curve.float()
    base = c[0]
    final = c[-1]
    span = (final - base)
    if span.abs() < 1e-8:
        return 1
    reached = (c - base) / span >= frac
    idx = int(torch.nonzero(reached, as_tuple=False)[0].item()) if reached.any() else len(c) - 1
    return idx + 1

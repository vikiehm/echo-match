import numpy as np
import torch
from utils.tensor_util import to_numpy
import scipy
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def nn_query(feat_x, feat_y, dim=-2):
    """
    Find correspondences via nearest neighbor query
    Args:
        feat_x: feature vector of shape x. [V1, C].
        feat_y: feature vector of shape y. [V2, C].
        dim: number of dimension
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2].
    """
    dist = torch.cdist(feat_x, feat_y)  # [V1, V2]
    p2p = dist.argmin(dim=dim)
    return p2p

def pointmap2fmap(p2p, evecs_x, evecs_y):
    """
    Compute functional map from point-to-point map
    Arg:
        p2p: point-to-point map (shape y -> shape x). [V2]
    Return:
        Cxy: functional map (shape x -> shape y). Shape [K, K]
    """
    evecs_x_a = evecs_x[p2p]
    evecs_y_a = evecs_y

    Cxy = torch.linalg.lstsq(evecs_y_a, evecs_x_a).solution
    return Cxy

def _fmap2pointmap(Cxy, evecs_x, evecs_y):
    """
    helper function to convert functional map to point-to-point map
    """
    dataA = evecs_x @ Cxy.t()
    dataB = evecs_y

    return dataA, dataB

def fmap2pointmap(Cxy, evecs_x, evecs_y):
    """
    Convert functional map to point-to-point map

    Args:
        Cxy: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    dataA, dataB = _fmap2pointmap(Cxy, evecs_x, evecs_y)
    p2p = nn_query(dataA, dataB)
    return p2p

def corr2fmap(corr_x, corr_y, evecs_x, evecs_y):
    """
    Compute functional map from correspondences
    Cxy : shape x -> shape y
    """
    evecs_x_a = evecs_x[corr_x]
    evecs_y_a = evecs_y[corr_y]

    Cxy = torch.linalg.lstsq(evecs_y_a, evecs_x_a).solution
    return Cxy

def get_mask(evals1, evals2, gamma=0.5, device="cpu"):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1.to(device) / scaling_factor, evals2.to(device) / scaling_factor
    evals_gamma1, evals_gamma2 = (evals1 ** gamma)[None, :], (evals2 ** gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()

def regularized_fmap_solve(A, B, evals_x, evals_y, lambda_=1e-3, resolvant_gamma=0.5):
    """Batched Inputs
    Compute the functional map matrix C by solving:
    min_C ||CA - B||^2 + lambda * ||D * C||^2
    where D is a mask based on eigenvalues that provides regularization.
    This is the same row-by-row solver implementation from DPFM and GeomFmaps papers.
    Bathed Output
    """
    D = get_mask(evals_x.flatten(), evals_y.flatten(), resolvant_gamma, A.device).unsqueeze(0)  # [B, K, K]

    A_t = A.transpose(1, 2)  # [B, C, K]
    A_A_t = torch.bmm(A, A_t)  # [B, K, K]
    B_A_t = torch.bmm(B, A_t)  # [B, K, K]
    
    C_i = []
    for i in range(evals_x.shape[1]):
        D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.shape[0])], dim=0)
        C = torch.bmm(torch.inverse(A_A_t + lambda_ * D_i), B_A_t[:, [i], :].transpose(1, 2))
        C_i.append(C.transpose(1, 2))
    C = torch.cat(C_i, dim=1)
    return C

def get_gt_fmap_with_regularization(corr_x, corr_y, evecs_x, evecs_y, evals_x, evals_y, lambda_=1e-3, resolvant_gamma=0.5, device="cuda"):
    """
    Un-Batched Inputs
    The functional map C is computed by solving:
    1. C = evecs_trans_y @ P @ evecs_x with additional regularization
    2. Minimizing: || evecs_y @ C = P @ evecs_x ||
    3. Equivalent to: || C.T @ evecs_y_a.T = evecs_x_a.T ||
    4. Final form: min_C || C.T @ coeff_Y = coeff_X ||
    Where:
    - P is the correspondence matrix
    - evecs_x, evecs_y are eigenvectors
    - coeff_Y, coeff_X are spectral coefficients
    Batched Output
    """

    # Un-Batched
    assert len(corr_x.shape) == 1 and len(corr_y.shape) == 1
    assert len(evecs_x.shape) == 2 and len(evecs_y.shape) == 2
    assert len(evals_x.shape) == 1 and len(evals_y.shape) == 1

    Nx, Ny = evecs_x.shape[0], evecs_y.shape[0]
    P = torch.zeros((Ny, Nx)).to(device)
    P[corr_y, corr_x] = 1

    evecs_x_a = P @ evecs_x
    evecs_y_a = evecs_y 
    coeff_X = evecs_x_a.t()
    coeff_Y = evecs_y_a.t()

    # Batched
    evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)
    coeff_X, coeff_Y = coeff_X.unsqueeze(0), coeff_Y.unsqueeze(0)
    C_transpose = regularized_fmap_solve(coeff_Y, coeff_X, evals_y, evals_x, lambda_, resolvant_gamma)

    C = C_transpose.transpose(1, 2)

    return C
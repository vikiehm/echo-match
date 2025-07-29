import torch
from utils.cache_util import get_cached_compute
def get_2d_isomap_emb(verts, faces, distances, cache_dir=None):
    return get_cached_compute(compute_isomap_embedding, verts, faces, dist=distances, dim=2, cache_dir=cache_dir)

def compute_c_matrix(d, h, m):
    return -1/(2*m) * h @ (d @ h)

def isomap(d: torch.Tensor, dim: int = 2) -> torch.Tensor:
    """
    Compute the ISOMAP embedding of points given their distance matrix.
    
    Parameters:
        d (torch.Tensor): Distance matrix between nodes. Should be square.
        dim (int): Number of dimensions to reduce to
    
    Returns:
        torch.Tensor: Embedded coordinates in reduced space
    """
    n, m = d.shape
    h = torch.eye(m, device=d.device) - (1/m)*torch.ones((m, m), device=d.device)
    d = d**2
    
    # Compute C matrix efficiently
    c = compute_c_matrix(d, h, m)
    
    # Use eigenvalues/vectors for symmetric matrix
    evals, evecs = torch.linalg.eigh(c)
    
    # Get largest eigenvalues/vectors
    evals = evals[-(dim):]
    evecs = evecs[:, -(dim):]
    
    # Sort in descending order
    evals = torch.flip(evals, dims=[0])
    evecs = torch.flip(evecs, dims=[1])
    
    z = evecs @ torch.diag(evals**(-1/2))
    return z

def compute_isomap_embedding(verts, faces, dist: torch.Tensor, dim: int = 2) -> torch.Tensor:
    """
    Get normalized ISOMAP embedding.
    
    Parameters:
        dist (torch.Tensor): Distance matrix
        dim (int): Target dimensionality
        
    Returns:
        torch.Tensor: Normalized embedding in [0,1]
    """
    # Perform ISOMAP dimensionality reduction
    v_embedded = isomap(dist, dim=dim)

    # Normalize the embedding to [0,1]
    v_min = torch.min(v_embedded, dim=0)[0]
    v_max = torch.max(v_embedded, dim=0)[0]
    v_embedded = (v_embedded - v_min) / (v_max - v_min)

    return v_embedded


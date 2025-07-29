import torch
from torch import nn
import numpy as np
from umap import UMAP
from umap.umap_ import find_ab_params
from utils.cache_util import get_cached_compute

def get_2d_umap_emb(verts, faces, distances, cache_dir=None):
    """
    Get 2D embedding for a mesh, using caching if possible.
    
    Args:
        verts (torch.Tensor): Vertex positions [V, 3]
        faces (torch.Tensor): Face indices [F, 3]
        cache_dir (str, optional): Directory to cache results. Default None.
    Returns:
        torch.Tensor: 2D embedding coordinates
    """
    return get_cached_compute(optimize_umap_embedding, verts, faces, distances=distances, cache_dir=cache_dir)


def compute_edge_distances(vertices, faces, device):
    edges = np.vstack((faces[:, [0, 1]], 
                      faces[:, [1, 2]], 
                      faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return edges, edge_lengths

def optimize_umap_embedding(vertices, faces, distances, 
                          n_components=2,
                          n_epochs=500,
                          initial_lr=1.0,
                          final_lr=1e-1,
                          attractive_weight=200.0,
                          edge_weight=500.0,
                          min_dist=1.0,
                          n_neighbors=300):
    """
    Optimize UMAP embedding with edge preservation.
    
    Args:
        vertices: np.array of shape (N, 3) - mesh vertices
        faces: np.array of shape (M, 3) - mesh faces
        distances: np.array of shape (N, N) - pairwise distances
        n_components: int - dimension of embedding
        n_epochs: int - number of optimization epochs
        initial_lr: float - initial learning rate
        final_lr: float - final learning rate
        attractive_weight: float - weight for attractive term
        edge_weight: float - weight for edge preservation
        min_dist: float - UMAP min_dist parameter
        n_neighbors: int - UMAP n_neighbors parameter
    
    Returns:
        np.array of shape (N, n_components) - optimized embedding
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get initial embedding from UMAP
    umap = UMAP(
        random_state=42,
        verbose=True,
        n_neighbors=n_neighbors,
        n_components=n_components,
        n_epochs=0,
        metric='precomputed'
    )
    init_embedding = umap.fit_transform(distances)
    graph = umap.graph_
    
    # Setup edge preservation
    edges, original_edge_lengths = compute_edge_distances(vertices, faces, device)
    edges_tensor = torch.tensor(edges, dtype=torch.long).to(device)
    edge_lengths_tensor = torch.tensor(original_edge_lengths, dtype=torch.float32).to(device)
    
    # Setup optimization
    graph_torch = torch.tensor(graph.todense(), dtype=torch.float32).to(device)
    embedding = nn.Parameter(torch.from_numpy(init_embedding).to(device, dtype=torch.float32))
    optimizer = torch.optim.AdamW([embedding], lr=initial_lr)
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma=(final_lr/initial_lr)**(1/n_epochs)
    )
    
    # Training loop
    for epoch in range(n_epochs):
        # Get UMAP parameters
        a, b = find_ab_params(1.0, min_dist)
        
        # Compute UMAP loss
        distance = torch.cdist(embedding, embedding, p=2)
        probability = -torch.log1p(a * distance ** (2 * b))
        log_prob = torch.nn.functional.logsigmoid(probability)
        
        attractive_term = -graph_torch * log_prob
        repulsive_term = -(1.0 - graph_torch) * (log_prob - probability)
        umap_loss = attractive_weight * attractive_term + repulsive_term
        
        # Compute edge preservation loss
        embedded_edge_lengths = torch.norm(
            embedding[edges_tensor[:, 0]]*0.15 - embedding[edges_tensor[:, 1]]*0.15, 
            dim=1
        )
        edge_loss = torch.mean((embedded_edge_lengths - edge_lengths_tensor) ** 2)
        
        # Total loss
        loss = torch.mean(umap_loss) + edge_weight * edge_loss
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # Get final embedding
    final_embedding = embedding.detach().cpu().numpy()
    
    # Normalize to [0,1]
    final_embedding = (final_embedding - np.min(final_embedding, axis=0)) / (np.max(final_embedding, axis=0) - np.min(final_embedding, axis=0))
    
    return final_embedding 
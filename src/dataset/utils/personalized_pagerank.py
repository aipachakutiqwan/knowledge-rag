import torch
import numpy as np
from torch_geometric.data.data import Data


def retrieval_via_pagerank(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5, alpha=0.85, max_iter=50, tol=1e-6):
    """
    Personalized PageRank-based retrieval with the same interface as retrieval_via_pcst.
    
    Args:
        graph: PyTorch Geometric Data object
        q_emb: Query embedding tensor
        textual_nodes: DataFrame with node attributes
        textual_edges: DataFrame with edge attributes
        topk: Number of top nodes to select based on PageRank scores
        topk_e: Number of top edges to consider (currently not used, kept for interface compatibility)
        cost_e: Edge cost parameter (currently not used, kept for interface compatibility)
        alpha: Damping factor for PageRank (default: 0.85)
        max_iter: Maximum number of PageRank iterations (default: 50)
        tol: Convergence tolerance for PageRank (default: 1e-6)
    
    Returns:
        data: Subgraph as PyTorch Geometric Data object
        desc: Description string with node and edge information
    """
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc
    
    num_nodes = graph.num_nodes
    device = graph.x.device
    
    # Compute cosine similarity between query and nodes for personalization vector
    node_similarities = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
    # Normalize to create personalization vector (must sum to 1)
    personalization = torch.softmax(node_similarities, dim=0)
    
    # Build adjacency matrix from edge_index
    edge_index = graph.edge_index
    
    # Compute out-degrees for normalization
    out_degree = torch.zeros(num_nodes, device=device)
    out_degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=device))
    # Avoid division by zero for isolated nodes
    out_degree = torch.clamp(out_degree, min=1.0)
    
    # Initialize PageRank vector
    pagerank = torch.ones(num_nodes, device=device) / num_nodes
    
    # Iterative PageRank computation: r = alpha * A * r + (1 - alpha) * personalization
    for _ in range(max_iter):
        pagerank_old = pagerank.clone()
        
        # Compute A * r (sparse matrix-vector multiplication)
        # For each edge (i, j), add r[i] / out_degree[i] to r_new[j]
        # Use scatter_add for efficient computation
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        contributions = pagerank[src_nodes] / out_degree[src_nodes]
        r_new = torch.zeros(num_nodes, device=device)
        r_new.scatter_add_(0, dst_nodes, contributions)
        
        # PageRank update: r = alpha * (A * r) + (1 - alpha) * personalization
        pagerank = alpha * r_new + (1 - alpha) * personalization
        
        # Check convergence
        if torch.norm(pagerank - pagerank_old) < tol:
            break
    
    # Select top-k nodes based on PageRank scores
    if topk > 0:
        topk = min(topk, num_nodes)
        _, topk_indices = torch.topk(pagerank, topk, largest=True)
        selected_nodes = topk_indices.cpu().numpy()
    else:
        # If topk is 0, select all nodes with non-zero PageRank
        selected_nodes = torch.nonzero(pagerank > tol, as_tuple=False).squeeze().cpu().numpy()
        if selected_nodes.ndim == 0:
            selected_nodes = np.array([selected_nodes.item()])
    
    # Select edges connected to selected nodes
    selected_nodes_tensor = torch.from_numpy(selected_nodes).to(device)
    # Create boolean masks for edges where src or dst is in selected_nodes
    src_mask = torch.isin(edge_index[0], selected_nodes_tensor)
    dst_mask = torch.isin(edge_index[1], selected_nodes_tensor)
    edge_mask = src_mask | dst_mask
    selected_edges = torch.nonzero(edge_mask, as_tuple=False).squeeze().cpu().numpy()
    if selected_edges.ndim == 0:
        selected_edges = np.array([selected_edges.item()])
    
    if len(selected_edges) == 0:
        # If no edges selected, return nodes only
        selected_edges = np.array([], dtype=np.int64)
        edge_index_selected = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        selected_edges = np.array(selected_edges)
        edge_index_selected = graph.edge_index[:, selected_edges]
    
    # Ensure all nodes in selected edges are included
    if len(selected_edges) > 0:
        all_nodes_in_edges = np.unique(np.concatenate([
            edge_index_selected[0].cpu().numpy(),
            edge_index_selected[1].cpu().numpy()
        ]))
        selected_nodes = np.unique(np.concatenate([selected_nodes, all_nodes_in_edges]))
    
    # Create textual description
    n = textual_nodes.iloc[selected_nodes]
    if len(selected_edges) > 0:
        e = textual_edges.iloc[selected_edges]
        desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
    else:
        desc = n.to_csv(index=False) + '\n' + textual_edges.iloc[:0].to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
    
    # Create node mapping for reindexing
    mapping = {int(n): i for i, n in enumerate(selected_nodes.tolist())}
    
    # Extract subgraph features
    x = graph.x[selected_nodes]
    if len(selected_edges) > 0:
        edge_attr = graph.edge_attr[selected_edges]
        src = [mapping[int(i)] for i in edge_index_selected[0].tolist()]
        dst = [mapping[int(i)] for i in edge_index_selected[1].tolist()]
        edge_index = torch.LongTensor([src, dst])
    else:
        edge_attr = torch.empty((0, graph.edge_attr.size(1)), device=device, dtype=graph.edge_attr.dtype)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))
    
    return data, desc

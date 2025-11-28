import torch
import numpy as np
from torch_geometric.data.data import Data
from collections import deque


def retrieval_via_k_hop(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5):
    """
    K-hop retrieval function that finds top-k seed nodes and expands k hops from them.
    
    Args:
        graph: Graph data with x, edge_index, edge_attr, num_nodes
        q_emb: Query embedding vector
        textual_nodes: DataFrame with textual node information
        textual_edges: DataFrame with textual edge information
        topk: Number of seed nodes to start from (default: 3)
        topk_e: Number of hops to expand (default: 3)
        cost_e: Not used in k-hop, kept for compatibility (default: 0.5)
    
    Returns:
        data: PyTorch Geometric Data object with subgraph
        desc: String description of nodes and edges in CSV format
    """
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    # Find top-k seed nodes based on cosine similarity to query
    if topk > 0:
        n_similarities = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_similarities, topk, largest=True)
        seed_nodes = topk_n_indices.cpu().numpy()
    else:
        seed_nodes = np.array([])

    # Perform k-hop expansion using BFS
    num_hops = max(0, int(topk_e)) if topk_e > 0 else 0
    
    if len(seed_nodes) == 0 or num_hops == 0:
        # No seed nodes or no hops, return empty subgraph
        selected_nodes = np.array([])
        selected_edges = np.array([], dtype=np.int64)
    else:
        # Build adjacency list for efficient neighbor lookup
        edge_index_np = graph.edge_index.cpu().numpy()
        adj_list = {i: set() for i in range(graph.num_nodes)}
        edge_to_idx = {}  # Map (src, dst) to edge index
        
        for idx, (src, dst) in enumerate(edge_index_np.T):
            adj_list[src].add(dst)
            adj_list[dst].add(src)  # Make it undirected
            edge_to_idx[(src, dst)] = idx
            edge_to_idx[(dst, src)] = idx  # Store reverse direction too
        
        # BFS expansion
        visited_nodes = set(seed_nodes.tolist())
        selected_edge_indices = set()
        queue = deque([(node, 0) for node in seed_nodes])  # (node, current_hop)
        
        while queue:
            current_node, current_hop = queue.popleft()
            
            if current_hop >= num_hops:
                continue
            
            # Add neighbors
            for neighbor in adj_list[current_node]:
                # Add edge to selected edges
                if (current_node, neighbor) in edge_to_idx:
                    edge_idx = edge_to_idx[(current_node, neighbor)]
                    selected_edge_indices.add(edge_idx)
                
                # Add neighbor node if not visited
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, current_hop + 1))
        
        selected_nodes = np.array(sorted(visited_nodes))
        selected_edges = np.array(sorted(selected_edge_indices))
        
        # Filter edges to only include those between selected nodes
        selected_nodes_set = set(selected_nodes)
        final_edge_indices = []
        
        for idx in selected_edges:
            src, dst = graph.edge_index[:, idx].cpu().numpy()
            if src in selected_nodes_set and dst in selected_nodes_set:
                final_edge_indices.append(idx)
        
        selected_edges = np.array(final_edge_indices) if final_edge_indices else np.array([], dtype=np.int64)
        
        # Ensure all nodes incident to selected edges are included
        if len(selected_edges) > 0:
            edge_index = graph.edge_index[:, selected_edges].cpu().numpy()
            incident_nodes = np.unique(np.concatenate([edge_index[0], edge_index[1]]))
            selected_nodes = np.unique(np.concatenate([selected_nodes, incident_nodes]))

    # Handle empty case
    if len(selected_nodes) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        return Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes), desc

    # Extract textual information
    n = textual_nodes.iloc[selected_nodes]
    if len(selected_edges) > 0:
        e = textual_edges.iloc[selected_edges]
        desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
    else:
        desc = n.to_csv(index=False) + '\n' + textual_edges.iloc[0:0].to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    # Create node mapping for relabeling
    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    # Build subgraph
    x = graph.x[selected_nodes]
    if len(selected_edges) > 0:
        edge_attr = graph.edge_attr[selected_edges]
        edge_index = graph.edge_index[:, selected_edges]
        src = [mapping[i] for i in edge_index[0].tolist()]
        dst = [mapping[i] for i in edge_index[1].tolist()]
        edge_index = torch.LongTensor([src, dst])
    else:
        # Create empty edge_attr with same shape as original
        if graph.num_edges > 0 and hasattr(graph.edge_attr, 'shape') and len(graph.edge_attr.shape) > 1:
            edge_attr = torch.empty(0, graph.edge_attr.shape[1], dtype=graph.edge_attr.dtype, device=graph.edge_attr.device)
        else:
            edge_attr = torch.empty(0, dtype=graph.edge_attr.dtype if graph.num_edges > 0 else torch.float32, device=graph.edge_attr.device if graph.num_edges > 0 else None)
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc


import torch
import numpy as np
from torch_geometric.data.data import Data
import functools
from .retrieval_with_prize_allocation import run_pcst
from .retrieval_func_selector import TeleportMode, PrizeAllocation


def retrieval_via_pagerank_fn(pcst, tele_mode, prize_allocation=None):
    return functools.partial(retrieval_via_pagerank, pcst=pcst, tele_mode=tele_mode, prize_allocation=prize_allocation)



def retrieval_via_pagerank(graph, q_emb, textual_nodes, textual_edges, pcst, tele_mode, prize_allocation, topk=3, topk_e=3, cost_e=0.5, alpha=0.85, max_iter=50, tol=1e-6):
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
        pcst: Whether to use PCST mode (required, no default)
        tele_mode: Teleport mode for PageRank (required, no default). Must be TeleportMode enum.
        prize_allocation: Prize allocation mode (required, no default). Must be PrizeAllocation enum or None.
                         None is only allowed when tele_mode=PROPORTIONAL and pcst=False.
                         When tele_mode=TOP_K or pcst=True, defaults to LINEAR if None.

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

    # Select top-k nodes based on similarity (used for both tele_mode and PCST)
    topk_similarity_indices = None
    if topk > 0:
        topk_for_similarity = min(topk, num_nodes)
        _, topk_similarity_indices = torch.topk(node_similarities, topk_for_similarity, largest=True)
    else:
        # If topk is 0, select all nodes with positive similarity
        topk_similarity_indices = torch.nonzero(node_similarities > 0, as_tuple=False).squeeze()
        if topk_similarity_indices.numel() == 0:
            topk_similarity_indices = None

    # Create personalization vector based on tele_mode
    personalization = torch.zeros(num_nodes, device=device)

    if tele_mode == TeleportMode.PROPORTIONAL:
        # Normalize to create personalization vector (must sum to 1)
        personalization = torch.softmax(node_similarities, dim=0)
    elif tele_mode == TeleportMode.TOP_K:
        # For TOP_K mode, use the pre-computed top-k similarity nodes
        if topk_similarity_indices is None:
            raise ValueError(f"topk must be > 0 when using tele_mode={tele_mode}, but got topk={topk}")

        topk_indices = topk_similarity_indices
        topk_for_tele = topk_indices.size(0)

        # Use prize_allocation to determine weights
        if prize_allocation == PrizeAllocation.LINEAR:
            # Linearly-proportional weights based on similarity ranking
            # Rank 1 (highest similarity) gets weight k, rank k gets weight 1
            weights = torch.arange(topk_for_tele, 0, -1, dtype=torch.float32, device=device)
        elif prize_allocation == PrizeAllocation.EQUAL:
            # Equal weighting for all top-k nodes
            weights = torch.ones(topk_for_tele, dtype=torch.float32, device=device)
        elif prize_allocation == PrizeAllocation.EXPONENTIAL:
            # Exponentially-decaying weights based on similarity ranking
            # Rank 1 gets highest weight, exponentially decreasing
            # Using exp(-0.5 * rank) where rank 0 (best) gets weight 1.0
            ranks = torch.arange(topk_for_tele, dtype=torch.float32, device=device)
            weights = torch.exp(-0.5 * ranks)
        else:
            raise ValueError(f"Unknown prize_allocation: {prize_allocation}. Must be one of {[mode.value for mode in PrizeAllocation]}")

        # Assign weights to top-k nodes
        personalization[topk_indices] = weights

        # Normalize to ensure personalization vector sums to 1
        personalization_sum = personalization.sum()
        if personalization_sum > 0:
            personalization = personalization / personalization_sum
        else:
            raise ValueError(f"Personalization vector sums to zero for tele_mode={tele_mode}. This should not happen.")
    else:
        raise ValueError(f"Unknown tele_mode: {tele_mode}. Must be one of {[mode.value for mode in TeleportMode]}")

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
        topk_clamped = min(topk, num_nodes)
        _, topk_indices = torch.topk(pagerank, topk_clamped, largest=True)
        topk_pagerank_nodes = topk_indices.cpu().numpy()
    else:
        # If topk is 0, select all nodes with non-zero PageRank
        topk_pagerank_nodes = torch.nonzero(pagerank > tol, as_tuple=False).squeeze().cpu().numpy()
        if topk_pagerank_nodes.ndim == 0:
            topk_pagerank_nodes = np.array([topk_pagerank_nodes.item()])

    if pcst:
        # PCST mode: union top-k nodes from PageRank and similarity, then run PCST

        # Use the pre-computed top-k similarity nodes
        if topk_similarity_indices is None:
            raise ValueError(f"topk must be > 0 when using pcst=True, but got topk={topk}")
        topk_similarity_nodes = topk_similarity_indices.cpu().numpy()
        if topk_similarity_nodes.ndim == 0:
            topk_similarity_nodes = np.array([topk_similarity_nodes.item()])

        # Union the two sets of nodes
        prize_nodes = np.unique(np.concatenate([topk_pagerank_nodes, topk_similarity_nodes]))

        # Set up node prizes: assign weights based on prize_allocation
        n_prizes = torch.zeros(num_nodes, device=device)
        if len(prize_nodes) > 0:
            # Get the similarity scores for prize nodes to determine ranking
            prize_nodes_tensor = torch.from_numpy(prize_nodes).to(device)
            prize_similarities = node_similarities[prize_nodes_tensor]
            # Sort by similarity (descending) to get ranking
            _, sorted_indices = torch.sort(prize_similarities, descending=True)
            
            # Assign prizes based on prize_allocation
            if prize_allocation == PrizeAllocation.LINEAR:
                # Linearly-proportional weights: rank 1 gets weight k, rank k gets weight 1
                prize_values = torch.arange(len(prize_nodes), 0, -1, dtype=torch.float32, device=device)
            elif prize_allocation == PrizeAllocation.EQUAL:
                # Equal weighting for all prize nodes
                prize_values = torch.ones(len(prize_nodes), dtype=torch.float32, device=device)
            elif prize_allocation == PrizeAllocation.EXPONENTIAL:
                # Exponentially-decaying weights
                ranks = torch.arange(len(prize_nodes), dtype=torch.float32, device=device)
                prize_values = torch.exp(-0.5 * ranks)
            else:
                raise ValueError(f"Unknown prize_allocation: {prize_allocation}. Must be one of {[mode.value for mode in PrizeAllocation]}")
            
            # Reorder prize_values according to similarity ranking
            prize_values_sorted = prize_values[sorted_indices]
            n_prizes[prize_nodes_tensor] = prize_values_sorted

        # Run PCST
        selected_nodes, selected_edges = run_pcst(graph, n_prizes, q_emb, topk_e, cost_e)

        if len(selected_edges) > 0:
            edge_index_selected = graph.edge_index[:, selected_edges]
        else:
            edge_index_selected = torch.empty((2, 0), dtype=torch.long, device=device)

    else:
        # Original behavior: use top-k nodes from PageRank
        selected_nodes = topk_pagerank_nodes

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

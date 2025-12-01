import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm

model_name = 'sbert'
path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'


class WebQSPDataset(Dataset):
    def __init__(self, sample_size: int, seed: int):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
        dataset, len_train, len_val, len_test, train_sample, val_sample, test_sample = sample_dataset(dataset, sample_size, seed)
        self.dataset = dataset
        self.q_embs = torch.load(f'{path}/q_embs.pt')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        graph = torch.load(f'{cached_graph}/{index}.pt', weights_only=False)
        desc = open(f'{cached_desc}/{index}.txt', 'r').read()
        label = ('|').join(data['answer']).lower()

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


def preprocess(sample_size: int, seed: int, retrieval_method: str):
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
    dataset, len_train, len_val, len_test, train_sample, val_sample, test_sample = sample_dataset(dataset, sample_size, seed)

    # Import the appropriate retrieval function based on method
    if retrieval_method == 'pcst':
        from src.dataset.utils.retrieval import retrieval_via_pcst as retrieval_func
    elif retrieval_method == 'k_hop':
        from src.dataset.utils.k_hop import retrieval_via_k_hop as retrieval_func
    elif retrieval_method == 'ppr':
        from src.dataset.utils.personalized_pagerank import retrieval_via_pagerank as retrieval_func
    else:
        raise ValueError(f"Unknown retrieval method: {retrieval_method}. Must be one of: 'pcst', 'k_hop', 'ppr'")

    q_embs = torch.load(f'{path}/q_embs.pt')
    for index in tqdm(range(len(dataset))):
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue

        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue
        graph = torch.load(f'{path_graphs}/{index}.pt', weights_only=False)
        q_emb = q_embs[index]
        subg, desc = retrieval_func(graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)
        torch.save(subg, f'{cached_graph}/{index}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)

def sample_dataset(dataset, sample_size: int, seed: int):
    np.random.seed(seed)
    train_size = min(sample_size, len(dataset['train']))
    val_size = min(sample_size, len(dataset['validation']))
    test_size = min(sample_size, len(dataset['test']))

    train_indices = np.random.choice(len(dataset['train']), size=train_size, replace=False)
    val_indices = np.random.choice(len(dataset['validation']), size=val_size, replace=False)
    test_indices = np.random.choice(len(dataset['test']), size=test_size, replace=False)

    train_sample = dataset['train'].select(train_indices)
    val_sample = dataset['validation'].select(val_indices)
    test_sample = dataset['test'].select(test_indices)
    len_train = len(train_sample)
    len_val = len(val_sample)
    len_test = len(test_sample)
    dataset = datasets.concatenate_datasets([train_sample, val_sample, test_sample])
    return dataset, len_train, len_val, len_test, train_sample, val_sample, test_sample

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Sample WebQSP dataset for training/inference.")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Number of examples to sample from each split (train/val/test). Must match preprocessing script.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for numpy sampling. Must match preprocessing script.",
    )
    parser.add_argument(
        "--retrieval_method",
        type=str,
        default="pcst",
        choices=["pcst", "k_hop", "ppr"],
        help="Retrieval method to use for subgraph extraction. Options: 'pcst', 'k_hop', 'ppr'.",
    )
    args = parser.parse_args()

    print(f"taking {args.sample_size} samples from each split")
    print(f"using seed {args.seed}")
    print(f"using retrieval method: {args.retrieval_method}")
    preprocess(args.sample_size, args.seed, args.retrieval_method)

    dataset = WebQSPDataset(args.sample_size, args.seed)

    data = dataset[1]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')

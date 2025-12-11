import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset, concatenate_datasets
from torch_geometric.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding


model_name = 'sbert'
path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'


def step_one(sample_size: int, seed: int):
    dataset = load_dataset("rmanluo/RoG-webqsp")
    print(f"dataset: {dataset}")
    dataset, len_train, len_val, len_test = sample_dataset(dataset, sample_size, seed)
    print(f"dataset concatenated: {dataset}")
    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        nodes = {}
        edges = []
        for tri in dataset[i]['graph']:
            h, r, t = tri
            h = h.lower()
            t = t.lower()
            if h not in nodes:
                nodes[h] = len(nodes)
            if t not in nodes:
                nodes[t] = len(nodes)
            edges.append({'src': nodes[h], 'edge_attr': r, 'dst': nodes[t]})
        nodes = pd.DataFrame([{'node_id': v, 'node_attr': k} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
        edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

        nodes.to_csv(f'{path_nodes}/{i}.csv', index=False)
        edges.to_csv(f'{path_edges}/{i}.csv', index=False)


def generate_split(sample_size: int, seed: int):

    dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset, len_train, len_val, len_test = sample_dataset(dataset, sample_size, seed)
    train_indices = np.arange(len_train)
    val_indices = np.arange(len_val) + len_train
    test_indices = np.arange(len_test) + len_train + len_val

    # Fix bug: remove the indices of the empty graphs from the val indices
    val_indices = [i for i in val_indices if i != 2937]

    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    os.makedirs(f'{path}/split', exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/split/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/split/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/split/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))


def step_two(sample_size: int, seed: int):
    print('Loading dataset...')
    dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset, len_train, len_val, len_test = sample_dataset(dataset, sample_size, seed)
    questions = [i['question'] for i in dataset]

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    # encode questions
    print('Encoding questions...')
    q_embs = text2embedding(model, tokenizer, device, questions)
    torch.save(q_embs, f'{path}/q_embs.pt')

    print('Encoding graphs...')
    os.makedirs(path_graphs, exist_ok=True)
    for index in tqdm(range(len(dataset))):

        # nodes
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        nodes.node_attr.fillna("", inplace=True)
        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())

        # edges
        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
        torch.save(pyg_graph, f'{path_graphs}/{index}.pt')

def sample_dataset(dataset, sample_size: int, seed: int):
    np.random.seed(seed)
    train_size = min(1, int(sample_size / 12 * 5), len(dataset['train']))
    val_size = min(1, int(sample_size / 12 * 3), len(dataset['validation']))
    test_size = min(1, int(sample_size / 12 * 4), len(dataset['test']))

    train_indices = np.random.choice(len(dataset['train']), size=train_size, replace=False)
    val_indices = np.random.choice(len(dataset['validation']), size=val_size, replace=False)
    test_indices = np.random.choice(len(dataset['test']), size=test_size, replace=False)

    train_sample = dataset['train'].select(train_indices)
    val_sample = dataset['validation'].select(val_indices)
    test_sample = dataset['test'].select(test_indices)
    len_train = len(train_sample)
    len_val = len(val_sample)
    len_test = len(test_sample)
    dataset = concatenate_datasets([train_sample, val_sample, test_sample])
    return dataset, len_train, len_val, len_test


def parse_args():
    parser = argparse.ArgumentParser(description="Sample WebQSP dataset and build graph data.")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=600,
        help="Number of total samples to sample from the dataset. Must match dataset/webqsp_sample.py.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for numpy sampling. Must match dataset/webqsp_sample.py.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"taking {args.sample_size} samples from each split")
    print(f"using seed {args.seed}")
    step_one(args.sample_size, args.seed)
    step_two(args.sample_size, args.seed)
    generate_split(args.sample_size, args.seed)

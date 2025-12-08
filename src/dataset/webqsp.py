import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.dataset.prompts.webqsp_template import PromptTemplates

model_name = 'sbert'
path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'


class WebQSPDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = PromptTemplates.system_instruction
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
        self.dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
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


def preprocess(retrieval_method: str, tele_mode: str = None, pcst: bool = False, prize_allocation: str = None):
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    # Get the appropriate retrieval function
    from src.dataset.utils.retrieval_func_selector import get_retrieval_func
    retrieval_func = get_retrieval_func(retrieval_method, tele_mode, pcst, prize_allocation)

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


if __name__ == '__main__':
    import argparse
    from src.dataset.utils.retrieval_func_selector import generate_extra_annotation

    parser = argparse.ArgumentParser(description="Preprocess WebQSP dataset for training/inference.")
    parser.add_argument(
        "--retrieval_method",
        type=str,
        default="pcst",
        choices=["pcst", "k_hop", "ppr"],
        help="Retrieval method to use for subgraph extraction. Options: 'pcst', 'k_hop', 'ppr'.",
    )
    parser.add_argument(
        "--tele_mode",
        type=str,
        default="proportional",
        choices=["proportional", "top_k"],
        help="Teleport mode for PPR retrieval. Only used when retrieval_method='ppr'. Options: 'proportional', 'top_k'. Default: 'proportional'.",
    )
    parser.add_argument(
        "--pcst",
        action="store_true",
        help="Use PCST mode for PPR retrieval. Only used when retrieval_method='ppr'. Default: False.",
    )
    parser.add_argument(
        "--prize_allocation",
        type=str,
        default="linear",
        choices=["linear", "equal", "exponential"],
        help="Prize allocation mode. Used when retrieval_method='pcst' or when retrieval_method='ppr' with tele_mode='top_k' or when retrieval_method='ppr' with pcst=True. Options: 'linear', 'equal', 'exponential'. Default: 'linear'.",
    )
    args = parser.parse_args()

    print(f"using retrieval method: {args.retrieval_method}")
    if args.retrieval_method == 'ppr':
        print(f"using tele_mode: {args.tele_mode}")
        print(f"using pcst: {args.pcst}")
    if args.retrieval_method == 'pcst' or (args.retrieval_method == 'ppr' and (args.tele_mode == 'top_k' or args.pcst)):
        print(f"using prize_allocation: {args.prize_allocation or 'linear (default)'}")

    preprocess(args.retrieval_method, args.tele_mode, args.pcst, args.prize_allocation)

    dataset = WebQSPDataset()

    data = dataset[1]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
    
    # Generate and print the extra_annotation string for use in train.py and inference.py
    extra_annotation = generate_extra_annotation(
        args.retrieval_method,
        args.tele_mode,
        args.pcst,
        args.prize_allocation
    )
    print(f"\n{'='*80}")
    print("For bash/terminal, use the following:")
    print(f'  extra_annotation="{extra_annotation}"')
    print(f"\nThen use it in your commands:")
    print(f'  python train.py --extra_annotation $extra_annotation')
    print(f'  python inference.py --extra_annotation $extra_annotation')
    print(f"\nFor Colab/Jupyter notebooks, use:")
    print(f'  extra_annotation = "{extra_annotation}"')
    print(f"\nThen use it in your commands:")
    print(f'  !python train.py --extra_annotation {{{extra_annotation}}}')
    print(f'  !python inference.py --extra_annotation {{{extra_annotation}}}')
    print(f"{'='*80}")

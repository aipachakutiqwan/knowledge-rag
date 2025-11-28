import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding
from sklearn.model_selection import train_test_split


model_name = 'sbert'
path = 'dataset/scene_graphs'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'


def textualize_graph(data):
    # mapping from object id to index
    objectid2nodeid = {object_id: idx for idx, object_id in enumerate(data['objects'].keys())}
    nodes = []
    edges = []
    for objectid, object in data['objects'].items():
        # nodes
        node_attr = f'name: {object["name"]}'
        x, y, w, h = object['x'], object['y'], object['w'], object['h']
        if len(object['attributes']) > 0:
            node_attr = node_attr + '; attribute: ' + (', ').join(object["attributes"])
        node_attr += '; (x,y,w,h): ' + str((x, y, w, h))
        nodes.append({'node_id': objectid2nodeid[objectid], 'node_attr': node_attr})

        # edges
        for rel in object['relations']:
            src = objectid2nodeid[objectid]
            dst = objectid2nodeid[rel['object']]
            edge_attr = rel['name']
            edges.append({'src': src, 'edge_attr': edge_attr, 'dst': dst})

    return nodes, edges


def step_one(max_graphs):
    dataset = json.load(open('dataset/gqa/train_sceneGraphs.json'))

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)
    count = 0
    image_ids = []
    for imageid, object in tqdm(dataset.items(), total=len(dataset)):
        node_attr, edge_attr = textualize_graph(object)
        pd.DataFrame(node_attr, columns=['node_id', 'node_attr']).to_csv(f'{path_nodes}/{imageid}.csv', index=False)
        pd.DataFrame(edge_attr, columns=['src', 'edge_attr', 'dst']).to_csv(f'{path_edges}/{imageid}.csv', index=False)
        image_ids.append(int(imageid))
        count += 1
        if count >= max_graphs:
            break
    print(f"image_ids: {image_ids}")
    print(f"len of image_ids: {len(image_ids)}")
    return image_ids


def step_two(image_ids):
    def _encode_questions():
        q_embs = text2embedding(model, tokenizer, device, df.question.tolist())
        torch.save(q_embs, f'{path}/q_embs.pt')
        #print(f"q_embs: {q_embs}")

    def _encode_graphs():
        print(f"len df: {len(df)}")
        image_ids = df.image_id.unique()
        print(f"image_ids to encode: {image_ids}")
        print(f"len image_ids : {len(image_ids)}")
        
        for i in tqdm(image_ids):
            nodes = pd.read_csv(f'{path_nodes}/{i}.csv')
            edges = pd.read_csv(f'{path_edges}/{i}.csv')
            if len(nodes) == 0:
                print(f'Empty graph, skipping image id {i}')
                continue
            node_attr = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            edge_index = torch.tensor([edges.src, edges.dst]).long()
            pyg_graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
            torch.save(pyg_graph, f'{path_graphs}/{i}.pt')

    df_questions = pd.read_csv(f'{path}/questions.csv')
    print(f"df questions: {len(df_questions)}")
    df = df_questions[df_questions['image_id'].isin(image_ids)]
    print(f"len df questions filtered: {len(df)}")
    df.to_csv(f'{path}/questions_sample.csv', index=False)
    os.makedirs(path_graphs, exist_ok=True)
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    _encode_questions()
    _encode_graphs()
    return df


def generate_split():

    # Load the data
    path = "dataset/scene_graphs"
    questions = pd.read_csv(f"{path}/questions_sample.csv")
    print(f"len questions : {len(questions)}")

    # Create a unique list of image IDs
    unique_image_ids = questions['image_id'].unique()
    print(f"len unique_image_ids : {len(unique_image_ids)}")

    # Shuffle the image IDs
    np.random.seed(42)  # For reproducibility
    shuffled_image_ids = np.random.permutation(unique_image_ids)

    # Split the image IDs into train, validation, and test sets
    train_ids, temp_ids = train_test_split(shuffled_image_ids, test_size=0.4, random_state=42)  # 60% train, 40% temporary
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)  # Split the 40% into two 20% splits

    # Create a mapping from image ID to set label
    id_to_set = {image_id: 'train' for image_id in train_ids}
    id_to_set.update({image_id: 'val' for image_id in val_ids})
    id_to_set.update({image_id: 'test' for image_id in test_ids})

    print(f"id_to_set: {id_to_set}")

    # Map the sets back to the original DataFrame
    questions['set'] = questions['image_id'].map(id_to_set)

    print(f"{questions.head(5).to_string()}")

    # Create the final train, validation, and test DataFrames
    train_df = questions[questions['set'] == 'train']
    val_df = questions[questions['set'] == 'val']
    test_df = questions[questions['set'] == 'test']

    # Create a folder for the split
    os.makedirs(f'{path}/split', exist_ok=True)

    # Writing the indices to text files
    train_df.index.to_series().to_csv(f'{path}/split/train_indices.txt', index=False, header=False)
    val_df.index.to_series().to_csv(f'{path}/split/val_indices.txt', index=False, header=False)
    test_df.index.to_series().to_csv(f'{path}/split/test_indices.txt', index=False, header=False)


if __name__ == '__main__':
    max_graphs = 10
    image_ids = step_one(max_graphs)
    step_two(image_ids)
    generate_split()

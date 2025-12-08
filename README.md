# Chat with your Knowledge Graph

## 🤖 Introduction

This repository features a Graph RAG (Retrieval-Augmented Generation) project that allows users to query complex Knowledge Graphs using a natural language, conversational interface.

Designed for real-world, textual graphs, this flexible Question Answering framework is highly versatile and applicable across diverse domains, including:

- Scene Graph Understanding
- Common Sense Reasoning
- Advanced Knowledge Graph Reasoning

## ⚡ Ablative Analysis

Our research systematically investigates the G-Retriever architecture. The core focus is an ablation analysis to rigorously quantify how modifications to the system's components impact the overall effectiveness and performance of the Knowledge Graph conversational system.


## :rocket: Setup

### 🌱  Create an environment and install dependencies

Check out the **Managing dependencies** section of the Contributing guide to learn how to set up the environment and install dependencies.

### 🌱 Setting up env variables

You can use `.env` file for set the following enviromental variables.
```
export HF_TOKEN="SET_YOUR_KEY_HERE" # Hugging Face token
```


## ✅ How to replicate the ablative analysis

## 🔭 Data Preprocessing:
```
# expla_graphs
python -m src.dataset.preprocess.expla_graphs
python -m src.dataset.expla_graphs

# scene_graphs
python -m src.dataset.preprocess.scene_graphs
python -m src.dataset.scene_graphs

# webqsp
python -m src.dataset.preprocess.webqsp
python -m src.dataset.webqsp

```


## 🔭 Varying Subgraph Retrieval Methods: 
We implemented two subgraph retrieval methods, K-hop and Personalized PageRank (PPR), and evaluated their performance using the ExplaGraphs and WebQSP datasets.

```

```
## 🔭 Varying Subgraph Encoder Type:
We implemented two graph neural network (GNN) architectures for the G-retriever architecture, GraphSAGE and the Graph Isomorphism Network (GIN), and evaluated their performance using the ExplaGraphs and WebQSP benchmark datasets.

```

```
## 🔭 Varying LLM Models:
We benchmarked a set of large language models (LLMs) with similar feature dimensions to Llama-2–7b-hf, including Gemma-7b, Mistral-7B-v0.1, Qwen-8B, and Deepseek-Coder-6.7b-base. Evaluation was performed using two representative datasets, ExplaGraphs and WebQSP, along with their respective LLM-instructed variants.

```

```
## 🔭 Prompt Tuning:
We implemented a specific system prompt template and measured its effect on performance when applied to the WebQSP knowledge graph question answering dataset.

```

```
## 🔭 New Graph RAG Model Architecture:
We propose a novel model architecture that integrates prompt tuning and G-Retriever with LoRA-based Large Language Model (LLM) fine-tuning. We then measure its performance using the WebQSP knowledge graph question answering dataset.

```

```






## :pushpin: Version tools


python: 3.12
cuda: 12.4 (verify with nvidia-smi command)
torch: 2.6.0+cu124
torchvision: 0.21.0

## Environment setup using UV (Recommended - GPU)
```
uv venv --python 3.12
source .venv/bin/activate

uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.6.0+cu124.html
uv pip install torch_sparse -f https://pytorch-geometric.com/whl/torch-2.6.0+cu124.html
uv pip install pyg_lib torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

uv pip install -r pyproject.toml --group dev

```

## Environment setup using UV (Recommended - CPU local machine)
```
uv venv --python 3.12
source .venv/bin/activate

uv pip install torch torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
uv pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.6.0+cpu.html
uv pip install torch_sparse -f https://pytorch-geometric.com/whl/torch-2.6.0+cpu.html
uv pip install pyg_lib  torch_cluster torch_spline_conv -f https://pytorch-geometric.com/whl/torch-2.6.0+cpu.html
uv pip install -r pyproject.toml --group dev


```


## Commands for environment verification
```
ldd --version
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBC
nvcc --version
ls -l /usr/local | grep cuda
nvidia-smi
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```




## Data Preprocessing
```
# expla_graphs
python -m src.dataset.preprocess.expla_graphs
python -m src.dataset.expla_graphs

# scene_graphs, might take
python -m src.dataset.preprocess.scene_graphs
python -m src.dataset.scene_graphs

# webqsp
python -m src.dataset.preprocess.webqsp
python -m src.dataset.webqsp
```

## Training
Replace path to the llm checkpoints in the `src/model/__init__.py`, then run

### 1) Inference-Only LLM
```
python inference.py --dataset scene_graphs --model_name inference_llm --llm_model_name 7b_chat

python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name 7b_chat


```
### 2) Frozen LLM + Prompt Tuning
```
# prompt tuning
python train.py --dataset scene_graphs_baseline --model_name pt_llm

python train.py --dataset expla_graphs --model_name pt_llm


# G-Retriever
python train.py --dataset scene_graphs --model_name graph_llm
```

### 3) Tuned LLM
```
# finetune LLM with LoRA
python train.py --dataset scene_graphs_baseline --model_name llm --llm_frozen False

# G-Retriever with LoRA
python train.py --dataset scene_graphs --model_name graph_llm --llm_frozen False
```

## Reproducibility
Use `run.sh` to run the codes and reproduce the published results in the main table.


## Samples retrieval graphs 

python -m src.dataset.preprocess.scene_graphs_sample
python -m src.dataset.scene_graphs_sample

python -m src.dataset.preprocess.webqsp_sample
python -m src.dataset.webqsp_sample

python -m src.dataset.preprocess.expla_graphs
python -m src.dataset.expla_graphs
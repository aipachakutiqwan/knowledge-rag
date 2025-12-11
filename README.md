# Chat with your Knowledge Graph

## 🤖 Introduction

This repository features a Graph RAG (Retrieval-Augmented Generation) project that allows users to query complex Knowledge Graphs using a natural language, conversational interface.

Designed for real-world, textual graphs, this flexible Question Answering framework is highly versatile and applicable across diverse domains, including:

- Scene Graph Understanding
- Common Sense Reasoning
- Advanced Knowledge Graph Reasoning

## ⚡ Ablative Analysis

Our research systematically investigates the G-Retriever architecture [![arXiv](https://img.shields.io/badge/arXiv-2402.07630-b31b1b.svg)](https://arxiv.org/abs/2402.07630). 
The core focus is an ablation analysis to rigorously quantify how modifications to the system's components impact the overall effectiveness and performance of the Knowledge Graph conversational system. 


## :rocket: Setup

### 🌱  Create an environment and install dependencies

Check out the **Managing dependencies** section of the **Contributing** guide to learn how to set up the environment and install dependencies.

### 🌱 Setting up env variables

You can use `.env` file for set the following enviromental variables. 

Create account in [Hugging Face](https://huggingface.co/settings/tokens) and get your token from [here](https://huggingface.co/settings/tokens).
```
export HF_TOKEN=<your_api_key>
```
Create account in  [W&B](https://wandb.ai/site) to track, visualize and manage the ablative analysis. To authenticate your machine with W&B, generate an API key from your user profile or at [here](https://wandb.ai/authorize).

```
export WANDB_API_KEY=<your_api_key>
```

## :pushpin: Version tools

All end-to-end testing and implementation of this project were conducted on a system featuring **dual A100 80GB GPUs** with the exception of the **ablation analysis for the subgraph retrieval methods**, which was conducted on a single A100 80GB GPU. To ensure reproducible results for the ablative analysis, this **hardware configuration must be utilized**.

Key software versions used:

- Python: 3.12
- Cuda: 12.4 
- Torch: 2.6.0+cu124
- Torchvision: 0.21.0

Use the following commands to confirm your installed versions match the required dependencies:
```
nvidia-smi
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```


## ✅ How to replicate the ablative analysis

Each component of the ablative analysis is presented in its own dedicated section. For full reproducibility, we include:

- **A brief description of the experiment's goal**.
- **The specific command/instructions needed for execution**.
- **A Weights & Biases link to access the live metrics, logs, and definitive results**.


### 🔭 Varying Subgraph Retrieval Methods:

We conducted an ablative analysis to compare the performance of three subgraph retrieval methods: K-hop, Personalized PageRank (PPR), and PPR + Prize-Collecting Steiner Tree (PCST).

Evaluation was performed using a sampled subset of the WebQSP dataset. It is important to note that, due to the sampling, the absolute performance metrics are subject to underfitting compared to results obtained on the full dataset. Our primary objective, however, is to provide a relative comparison of the retrieval methods.

The analysis was executed on a **single NVIDIA A100 80GB GPU instance**, accessible via a Colab environment.

The Colab notebook is available here: 
<table align="center">
  <td>
    <a target="_blank" href="https://colab.research.google.com/drive/1rHoSkz0eV-ACvFIPXXObyZsLKcyxbrPe"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

- [WebQSP - Subgraph Retrieval Ablation](https://api.wandb.ai/links/florenciopaucar-uni/we6r7usf)


### 🔭 Data Preprocessing:

Prior to running the ablative analysis, the dataset requires preprocessing. This essential step includes textualizing the graph nodes and edges, embedding the questions, and then saving the resulting graphs in the PyTorch Geometric Data format. Finally, the dataset must be correctly split into train, validation, and test sets. Use the below commands to preprocess the datasets.

```
## ExplaGraphs
python -m src.dataset.preprocess.expla_graphs
python -m src.dataset.expla_graphs

## SceneGraphs
python -m src.dataset.preprocess.scene_graphs

## WebQSP
python -m src.dataset.preprocess.webqsp

```


### 🔭 Varying Subgraph Encoder Type:
We implemented two graph neural network (GNN) architectures for the G-retriever architecture, GraphSAGE and the Graph Isomorphism Network (GIN), and evaluated their performance using the ExplaGraphs and WebQSP benchmark datasets.

```
## GraphSAGE
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False  --gnn_model_name graphsage
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False  --gnn_model_name graphsage

## GIN
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False  --gnn_model_name gin
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False  --gnn_model_name gin

```
- [GraphSAGE benchmarking for ExplaGraphs](https://wandb.ai/florenciopaucar-uni/project_g_retriever/reports/GraphSAGE-ExplaGraphs--VmlldzoxNTI0ODY2Nw?accessToken=t7s1tqhv0wru55i1jbp9dzatl2gdexjg80pupj0wa9qijvga7bwc3onmfsdeusba)
- [GraphSAGE benchmarking for WebQSP](https://wandb.ai/florenciopaucar-uni/project_g_retriever/reports/GraphSAGE-WebQSP--VmlldzoxNTI0ODk2OA?accessToken=rl84ivm79vt5cufdob8ajvne7ecrjpwlxardi18u8uzqc1ugo75uoccguuntbef9)
- [GIN benchmarking for ExplaGraphs](https://wandb.ai/florenciopaucar-uni/project_g_retriever/reports/GIN-ExplaGraphs--VmlldzoxNTI0ODY5MQ?accessToken=9i50yn52010qz3zwurlj9wjuhenucgy3gm1x1479xrrqpataq96rqbjtc2punoab)
- [GIN benchmarking for WebQSP](https://wandb.ai/florenciopaucar-uni/project_g_retriever/reports/GIN-WebQSP--VmlldzoxNTI1OTYwNg?accessToken=shf3cmrptkodzfsfskwpuv7au6vfz4n35n7e3l4w59y03982yu4xmfgqujgx05uq)


### 🔭 Varying LLM Models:
We benchmarked a set of large language models (LLMs) with similar feature dimensions to Llama-2–7b-hf, including Gemma-7b, Mistral-7B-v0.1, Qwen-8B, and Deepseek-Coder-6.7b-base. Evaluation was performed using two representative datasets, ExplaGraphs and WebQSP, along with their respective LLM-instructed variants.

```
## LLM model comparisons
### Gemma
#### a) Tuned llm: g-retriever + finetuning with lora
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False --llm_model_name gemma_7b
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False --llm_model_name gemma_7b

### Mistral
#### a) Inference only: Question-Only
python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name mistral_7b_it --max_txt_len 0
#### b) Tuned llm: g-retriever + finetuning with lora
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False --llm_model_name mistral_7b
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False --llm_model_name mistral_7b

### Qwen
#### a) Inference only: Question-Only
python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name qwen3_8b_it --max_txt_len 0
#### b) Tuned llm: g-retriever + finetuning with lora
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False --llm_model_name qwen3_8b
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False --llm_model_name qwen3_8b

### DeepSeek
#### a) Inference only: Question-Only
python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name deepseek6_7b_it --max_txt_len 0
#### b) Tuned llm: g-retriever + finetuning with lora
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False --llm_model_name deepseek6_7b
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False --llm_model_name deepseek6_7b

```
- [ExplaGraphs Inference only in Instructed LLMs](https://wandb.ai/florenciopaucar-uni/project_g_retriever/reports/ExplaGraphs-LLMs-Inference-Only--VmlldzoxNTI3MzYwNA?accessToken=cf2k9lbvlaz8m7l7pm7j9vs3ku0f62chk94eotd2uy5j3ufqw82wg28uynyrnam8)
- [Diverse LLM Tuning: G-retriever + finetuning with Lora in ExplaGraphs](https://wandb.ai/florenciopaucar-uni/project_g_retriever/reports/ExplaGraphs-LLMs-G-Retriever-Lora--VmlldzoxNTI3MzUwMw?accessToken=bge6arszopkqssgseocg4mpu74nkbs1tgw2pzwzyeohdgltm03qm1xo82hgy6inm)
- [Diverse LLM Tuning: G-retriever + finetuning with Lora in WebQSP](https://wandb.ai/florenciopaucar-uni/project_g_retriever/reports/WebQSP-LLMs-G-Retriever-Lora--VmlldzoxNTI4MjIyMQ?accessToken=pizj2pshzwmc8q87xgdkqwawn1qn7ldmczewnwaxwyi80tu4jk6809v5q7t0cwmx)


### 🔭 New Graph RAG Model configuration:
We propose a novel model configuration that integrates prompt tuning and G-Retriever with LoRA-based Large Language Model (LLM) fine-tuning. We then measure its performance using the WebQSP knowledge graph question answering dataset.

```
## New Graph RAG Model configuration
### Tuned llm: g-retriever + finetuning with lora + prompt tuning: Fine-tunning the LLM with LoRA and prompt tuning
python train.py --dataset webqsp --model_name graph_llm_pt --llm_frozen False 

```
- [GraphLLMPromptTuning model using WebQSP](https://wandb.ai/florenciopaucar-uni/project_g_retriever/reports/WebQSP-Prompt-Tuning-G-Retriever-LLM-LoRA--VmlldzoxNTI4NTM4NQ?accessToken=y16u59kzx8o5335rwzjqsjfr6bb7zfstfc044w8m8vxeaiai7q4ms3krft1jp3u9)



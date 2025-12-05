# ExplaGraphs

## 1) inference only: Using LLM for direct question answering
### a) Question-Only
python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 0
### b) Textual Graph + Question
python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name 7b_chat

## 2) frozen llm + prompt tuning: Keeping the parameters of the LLM frozen and adapting only the prompt.
### a) prompt tuning
python train.py --dataset expla_graphs --model_name pt_llm
### b) g-retriever
python train.py --dataset expla_graphs --model_name graph_llm

## 3) tuned llm: Fine-tunning the LLM with LoRA
### a) finetuning with lora
python train.py --dataset expla_graphs --model_name llm --llm_frozen False
### b) g-retriever + finetuning with lora
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False

# Scene Graphs

## 1) inference only: Using LLM for direct question answering
### a) Question-Only
python inference.py --dataset scene_graphs --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 0  
### b) Textual Graph + Question
python inference.py --dataset scene_graphs --model_name inference_llm --llm_model_name 7b_chat  

## 2) frozen llm + prompt tuning: Keeping the parameters of the LLM frozen and adapting only the prompt.
### a) prompt tuning
python train.py --dataset scene_graphs --model_name pt_llm 
### b) g-retriever
python train.py --dataset scene_graphs --model_name graph_llm 

## 3) tuned llm: Fine-tunning the LLM with LoRA
### a) finetuning with lora
python train.py --dataset scene_graphs_baseline --model_name llm --llm_frozen False  
### b) g-retriever + finetuning with lora
python train.py --dataset scene_graphs --model_name graph_llm --llm_frozen False 

# WebQSP

## 1) inference only: Using LLM for direct question answering
### a) Question-Only
python inference.py --dataset webqsp --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 0  
### b) Textual Graph + Question
python inference.py --dataset webqsp --model_name inference_llm --llm_model_name 7b_chat  

## 2) frozen llm + prompt tuning: Keeping the parameters of the LLM frozen and adapting only the prompt.
### a) prompt tuning
python train.py --dataset webqsp --model_name pt_llm  
### b) g-retriever
python train.py --dataset webqsp --model_name graph_llm  

## 3) tuned llm: Fine-tunning the LLM with LoRA
### a) finetuning with lora
python train.py --dataset webqsp_baseline --model_name llm --llm_frozen False  
### b) g-retriever + finetuning with lora
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False 


## GNN Architectures
### GraphSAGE
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False  --gnn_model_name graphsage
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False  --gnn_model_name graphsage

### GIN
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False  --gnn_model_name gin
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False  --gnn_model_name gin

## LLM model comparisons
### Gemma
#### a) inference only: Question-Only
python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name gemma_7b_it --max_txt_len 0
#### b) tuned llm: g-retriever + finetuning with lora
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False --llm_model_name gemma_7b
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False --llm_model_name gemma_7b

### Mistral
#### a) inference only: Question-Only
python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name mistral_7b_it --max_txt_len 0
#### b) tuned llm: g-retriever + finetuning with lora
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False --llm_model_name mistral_7b
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False --llm_model_name mistral_7b

### Qwen
#### a) inference only: Question-Only
python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name qwen3_8b_it --max_txt_len 0
#### b) tuned llm: g-retriever + finetuning with lora
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False --llm_model_name qwen3_8b
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False --llm_model_name qwen3_8b

### DeepSeek
#### a) inference only: Question-Only
python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name deepseek6_7b_it --max_txt_len 0
#### b) tuned llm: g-retriever + finetuning with lora
python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False --llm_model_name deepseek6_7b
python train.py --dataset webqsp --model_name graph_llm --llm_frozen False --llm_model_name deepseek6_7b

### LLM Prompt Templates

## frozen llm + prompt tuning: prompt tuning: Keeping the parameters of the LLM frozen and adapting only the prompt.
python train.py --dataset webqsp --model_name pt_llm  


## tuned llm: g-retriever + finetuning with lora + prompt tuning: Fine-tunning the LLM with LoRA and prompt tuning
python train.py --dataset webqsp --model_name graph_llm_pt --llm_frozen False 


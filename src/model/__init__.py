from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.graph_llm import GraphLLM
from src.model.graph_llm_pt import GraphLLMPromptTuning

load_model = {
    "llm": LLM,
    "inference_llm": LLM,
    "pt_llm": PromptTuningLLM,
    "graph_llm": GraphLLM,
    "graph_llm_pt": GraphLLMPromptTuning,
}

# Replace the following with the model paths
dict_llm_model_path = {
    "1b": "meta-llama/Llama-3.2-1B",
    "7b": "meta-llama/Llama-2-7b-hf",
    "7b_chat": "meta-llama/Llama-2-7b-chat-hf",
    "13b": "meta-llama/Llama-2-13b-hf",
    "13b_chat": "meta-llama/Llama-2-13b-chat-hf",
    "gemma_7b": "google/gemma-7b",
    "gemma_7b_it": "google/gemma-7b-it",
    "mistral_7b": "mistralai/Mistral-7B-v0.1",
    "mistral_7b_it": "mistralai/Mistral-7B-Instruct-v0.1",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_8b_it": "Qwen/Qwen3-8B",
    "deepseek6_7b": "deepseek-ai/deepseek-coder-6.7b-base",
    "deepseek6_7b_it": "deepseek-ai/deepseek-coder-6.7b-instruct"
}

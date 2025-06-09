import subprocess
from huggingface_hub import login

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import os
from dotenv import load_dotenv

def run():        
    
    load_dotenv()
    token=os.getenv("token")
    model_id=os.getenv("model_id")
    path=os.getenv("path")
    device = 'cuda'


    login(token=token)    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True
    ).to("cuda")

    model.to(device)
    model.eval()

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",  f"{path}",
        "--tokenizer",  f"{path}",
        # "--model",  "./merged-prompt-injection-model",
        # "--tokenizer",  "./merged-prompt-injection-model",
        # "--gpu-memory-utilization", str(cfg.model.gpu_memory_utilization),         
        "--port", "7779",
        "--dtype", "float16"

    ]
    subprocess.run(command)


if __name__ == "__main__":
    run()
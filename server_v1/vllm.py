import subprocess
from huggingface_hub import login

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import os
from dotenv import load_dotenv

def run():        
    
    load_dotenv()
    token=os.getenv("token")
    model_id=os.getenv("model_id")
    path=os.getenv("path")

    login(token=token)    
    model_id = "r1char9/Oblivion2.5-1.5B-adapter"
    device = 'cuda'

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        device_map=device,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    model = PeftModel.from_pretrained(base_model, model_id, torch_dtype=base_model.dtype)
    model = model.merge_and_unload()
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
        "--dtype float16",


    ]
    subprocess.run(command)


if __name__ == "__main__":
    run()
import subprocess

import os
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def run():        
    model_id='Qwen/Qwen2.5-7B-Instruct'    
    save_path = './qwen2.5/'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",  "./qwen2.5/",
        "--tokenizer",  "./qwen2.5/",
        "--port", "7779",       
        "--tensor-parallel-size", "2",         
    ]
    subprocess.run(command)

if __name__ == "__main__":
    run()
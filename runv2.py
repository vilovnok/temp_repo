import subprocess
from huggingface_hub import login

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run():        
    
    login(token="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")    
    model_id = "../STAGE2-DEMO/checkpoint-14730/"
    # save_path = "./sft"
    
    # tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right", truncation=True)
    # model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", 
    #                                             torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda")

    # model.save_pretrained(save_path)
    # tokenizer.save_pretrained(save_path)

    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        # "--model",  "../STAGE2-DEMO/checkpoint-14730/",
        # "--tokenizer",  "../STAGE2-DEMO/checkpoint-14730/",
        "--model",  "../STAGE1-V1/checkpoint-4870/",
        "--tokenizer",  "../STAGE1-V1/checkpoint-4870/",
        "--port", "7779",
        # "--dtype", "float16 "
        # "--trust-remote-code"
    ]
    subprocess.run(command)

if __name__ == "__main__":
    run()
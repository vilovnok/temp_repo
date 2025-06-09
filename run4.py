import torch
import subprocess
from huggingface_hub import login

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig



def run():        
    login(token="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")    
    model_id = "Qwen/Qwen2.5-1.5B"
    adapter_id = "r1char9/Oblivion2.5-1.5B-v1"
    compute_dtype = getattr(torch, "float16")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config
    )
    model = PeftModel.from_pretrained(model, adapter_id)
    merged_model = model.merge_and_unload()
    save_path = "./stage1-v1"

    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",  "./stage1-v1",
        "--tokenizer",  "./stage1-v1",
        "--port", "7779",
        "--dtype float16"
        " --trust-remote-code"
    ]
    subprocess.run(command)


if __name__ == "__main__":
    run()

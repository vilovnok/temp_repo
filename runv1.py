import subprocess
from huggingface_hub import login

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def run():        
    
    login(token="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")    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", padding_side="right", truncation=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, "r1char9/adapter-prompt-2-prompt-injection")

    merged_model = model.merge_and_unload()
    save_path = "./merged-prompt-injection-model"
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",  "./merged-prompt-injection-model",
        "--tokenizer",  "./merged-prompt-injection-model",
        # "--gpu-memory-utilization", str(cfg.model.gpu_memory_utilization),         
        "--port", "7779",
        "--dtype float16"
        "--trust-remote-code"
    ]
    subprocess.run(command)


if __name__ == "__main__":
    run()
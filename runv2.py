import subprocess
from huggingface_hub import login

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def run():        
    
    login(token="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")    
    model_id = "r1char9/sft-prompt-2-prompt-injection"
    save_path = "./sft-prompt-injection-model"

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",  "./sft-prompt-injection-model",
        "--tokenizer",  "./sft-prompt-injection-model",
        "--port", "7779",
        "--dtype float16"
        "--trust-remote-code"
    ]
    subprocess.run(command)


if __name__ == "__main__":
    run()
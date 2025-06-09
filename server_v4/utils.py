import os
from dotenv import load_dotenv

load_dotenv()
system_prompt=os.getenv("system_prompt")

def format_chat(messages: list[dict]) -> str:
    out = ""
    for msg in messages:
        out += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return out

def build_prompt(user_query: str) -> str:
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User's request: {user_query}"},
    ]
    return messages


def build_prompt(user_query: str, system_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"User's request: {user_query}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

import os
import random
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
        "request with details",
        "request with consent",
        "request with links"
    ]
    prefix = random.choice(messages)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{prefix}: {user_query}"},
    ]
    return messages
    # return format_chat(messages)
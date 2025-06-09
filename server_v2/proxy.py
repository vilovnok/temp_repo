from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Optional
import httpx
import uvicorn

import os
from dotenv import load_dotenv
from .utils import build_prompt


load_dotenv()
url=os.getenv("url")


app = FastAPI()

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 1024

@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: ChatRequest):
    last_user_msg = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "")

    full_messages = build_prompt(last_user_msg)
    print(full_messages)
    payload = {
        "model": request.model,
        "messages": full_messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens
    }

    async with httpx.AsyncClient(timeout=360.0) as client:
        vllm_response = await client.post(url, json=payload)

    json = vllm_response.json()
    print('*'*100)
    print(json['choices'][0]['message']['content'])
    print('*'*100)
    return json

if __name__ == "__main__":
    uvicorn.run("server_v2.proxy:app", host="0.0.0.0", port=8000, reload=True)
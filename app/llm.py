from __future__ import annotations
from typing import List
from openai import OpenAI

class LLMClient:
    def __init__(self, api_key: str, chat_model: str, embed_model: str):
        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 800) -> str:
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()

    def embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.embed_model, input=text)
        return resp.data[0].embedding

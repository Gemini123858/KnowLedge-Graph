# api_utils.py

from typing import List, Tuple
import requests
from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from config import API_CONFIG, EMBEDDING_CONFIG
import logging

logging.basicConfig(level=logging.DEBUG)

class LocalEmbeddings(Embeddings):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        embeddings = []
        for text in texts:
            response = requests.post(url, json={"input": text, "model": self.model})
            embeddings.append(response.json()["data"][0]["apis"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        url = f"{self.base_url}/embeddings"
        response = requests.post(url, json={"input": text, "model": self.model})
        print(response.json())
        return response.json()["data"][0]["apis"]


def clean_api_response(response: str, api_type: str) -> str:
    """清理API响应"""
    if api_type == "DeepSeek":
        return response.replace("<｜end▁of▁sentence｜>", "").strip()
    return response.strip()


def test_api_connection(api_type: str, api_key: str, model_name: str) -> Tuple[bool, str]:
    """测试API连接"""
    try:
        base_url = API_CONFIG[api_type.lower()]['base_url']
        print(base_url)
        client = OpenAI(api_key = api_key, base_url=base_url)
        # client = OpenAI(api_key)
        print("client new")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hi!"}],
            # stream=True,
            max_tokens=10
        )
        print(response)
        raw_response = response.choices[0].message.content
        cleaned_response = clean_api_response(raw_response, api_type)
        return True, cleaned_response
    except Exception as e:
        return False, str(e)


def test_embeddings(embed_type: str, api_key: str = None, base_url: str = None, model: str = None) -> Tuple[bool, str]:
    """测试嵌入模型"""
    try:
        if embed_type == "本地":
            embeddings = LocalEmbeddings(
                base_url=base_url or EMBEDDING_CONFIG['local']['base_url'],
                model=model or EMBEDDING_CONFIG['local']['model']
            )
        else:
            embeddings = OpenAIEmbeddings(api_key=api_key)

        test_embedding = embeddings.embed_query("test")
        return True, f"成功生成嵌入向量，维度: {len(test_embedding)}"
    except Exception as e:
        return False, str(e)


def get_api_client(api_type: str, api_key: str, model_name: str) -> OpenAI:
    """获取API客户端"""
    base_url = API_CONFIG[api_type.lower()]['base_url']
    return OpenAI(api_key=api_key, base_url=base_url)

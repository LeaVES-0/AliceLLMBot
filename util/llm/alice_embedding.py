from logging import debug, info
from typing import Any, Dict, List, Callable
from bigdl.llm.models import Llama
from bigdl.llm.langchain.embeddings import TransformersEmbeddings

import numpy as np
from chromadb import EmbeddingFunction, Documents
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Extra


class AliceEmbedding(BaseModel, Embeddings):
    """AliceLLMEmbedding"""

    model: Any
    tokenizer: Any

    encode_kwargs: Dict[str, Any] = {}

    def __init__(self, model, tokenizer, **encode_kwargs):
        super().__init__()
        self.model = model.model
        self.tokenizer = tokenizer

        self.encode_kwargs = encode_kwargs

    class Config:
        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer models.
        :param texts: The list of texts to embed.
        :return List of embeddings.
        """
        texts = texts if texts else []
        doce = []

        for t in texts:
            t.replace("\n", " ")
            doce.append(t)
        embeddings = [self.model.embed(doc) for doc in doce]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a bigdl-llm transformer models.
        :param text: The text to embed.
        :return Embeddings for the text.
        """
        text = text.replace("\n", r" ")
        embedding = self.model.embed(text)
        return embedding


class AliceEmbeddingFunc(EmbeddingFunction):
    embedding: Embeddings

    def load(self, embedding: Embeddings):
        self.embedding = embedding

    def __call__(self, input_: Documents):
        r = self.embedding.embed_documents(texts=input_)
        return r

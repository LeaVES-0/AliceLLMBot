from typing import Any, Dict, List

import numpy as np
from langchain.embeddings.base import Embeddings
from langchain_core.language_models import LLM
from pydantic import BaseModel, Extra


class AliceEmbedding(BaseModel, Embeddings):
    """AliceLLMEmbedding"""

    model: Any
    tokenizer: Any

    encode_kwargs: Dict[str, Any] = {}

    def __init__(self, model, tokenizer, **encode_kwargs):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.encode_kwargs = encode_kwargs


    class Config:
        extra = Extra.forbid

    def embed(self, texts: str, **_):
        """Compute doc embeddings using a HuggingFace transformer model.
        :param texts: The list of texts to embed.
        :return st of embeddings, one for each text.
        """
        input_ids = self.tokenizer.encode(texts, return_tensors="pt")  # shape: [1, T]
        embeddings = self.model(input_ids, return_dict=False)[0]  # shape: [1, T, N]
        embeddings = embeddings.squeeze(0).detach().numpy()
        embeddings = np.mean(embeddings, axis=0)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.
        :param texts: The list of texts to embed.
        :return List of embeddings.
        """
        texts = []
        for t in texts:
            t.replace("\n", " ")
            texts.append(t)
        embeddings = [self.embed(text, **self.encode_kwargs).tolist() for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a bigdl-llm transformer model.
        :param text: The text to embed.
        :return Embeddings for the text.
        """
        text = text.replace("\n", r" ")
        embedding = self.embed(text, **self.encode_kwargs)
        return embedding.tolist()

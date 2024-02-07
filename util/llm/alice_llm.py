import os
from logging import Logger, getLogger
from typing import Any, Optional, List, Dict, Generator

import numpy as np
from bigdl.llm.models import Llama
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from pydantic import root_validator


class AliceLLM(LLM):
    os.environ['NUMEXPR_MAX_THREADS'] = '12'

    client: Any  # :meta private:

    model_path: str
    model_kwargs: Optional[dict] = None
    kwargs: Any

    logger: Logger = getLogger()

    lora_base: Optional[str] = None
    lora_path: Optional[str] = None
    n_ctx: int = 2048
    n_parts: int = -1
    seed: int = -1
    f16_kv: bool = True
    logits_all: bool = False
    vocab_only: bool = False
    use_mlock: bool = False
    n_threads: Optional[int] = 8
    n_batch: Optional[int] = 512
    n_gpu_layers: Optional[int] = 0
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[List[str]] = []
    repeat_penalty: Optional[float] = 1.1
    top_k: Optional[int] = 40
    last_n_tokens_size: Optional[int] = 64
    use_mmap: Optional[bool] = True

    streaming: bool = True

    @root_validator()
    def init(cls, values: Dict) -> Dict:
        model_path = values["model_path"]
        model_kwargs = values["model_kwargs"]
        kwargs = values["kwargs"]
        logger = values["logger"]
        model_param_names = [
            "lora_path",
            "lora_base",
            "n_ctx",
            "n_parts",
            "seed",
            "f16_kv",
            "logits_all",
            "vocab_only",
            "use_mlock",
            "n_threads",
            "n_batch",
            "use_mmap",
            "last_n_tokens_size",
        ]
        model_params = {k: values[k] for k in model_param_names}
        # For backwards compatibility, only include if non-null.
        if values["n_gpu_layers"] is not None:
            model_params["n_gpu_layers"] = values["n_gpu_layers"]

        try:
            values["client"] = Llama(model_path, **model_params)

        except Exception as e:
            raise e

        return values
    _default_params = {
        "suffix": suffix,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": logprobs,
        "echo": echo,
        "stop_sequences": stop,  # changed
        "repeat_penalty": repeat_penalty,
        "top_k": top_k,
    }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_path": self.model_path}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        return "AliceLLM"

    def _get_parameters(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:

        # Raise error if stop sequences are in both input and default params
        if self.stop and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")

        params = self._default_params

        print(params)
        # llama_cpp expects the "stop" key not this, so we remove it:
        try:
            params.pop("stop_sequences")
        except KeyError:
            pass

        # then sets it as configured, or default to an empty list:
        params["stop"] = self.stop or stop or []

        return params

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs
    ) -> str:
        if self.streaming:
            combined_text_output = ""
            for token in self.stream_(prompt=prompt, stop=stop, run_manager=run_manager):
                combined_text_output += token["choices"][0]["text"]
            return combined_text_output
        else:
            params = self._get_parameters(stop)
            result = self.client(prompt=prompt, **params)
            return result["choices"][0]["text"]

    def stream_(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Generator[Dict, None, None]:

        params = self._get_parameters(stop)
        result = self.client(prompt=prompt, stream=True, **params)
        for chunk in result:
            token = chunk["choices"][0]["text"]
            log_probs = chunk["choices"][0].get("logprobs", None)
            if run_manager:
                run_manager.on_llm_new_token(
                    token=token,
                    verbose=self.verbose,
                    log_probs=log_probs
                )
            yield chunk

    def get_num_tokens(self, text: str) -> int:
        return len(self.client.tokenize(text.encode("utf-8")))
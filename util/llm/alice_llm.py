import os
from logging import Logger, getLogger
from typing import Any, Optional, List, Dict, Generator

from bigdl.llm.models import Llama
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from pydantic import root_validator


class AliceLLM(LLM):
    os.environ['NUMEXPR_MAX_THREADS'] = '12'

    model: Any
    model_path: str
    model_kwargs: Optional[dict] = None

    logger: Logger = getLogger()

    # model keyword arguments
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[List[str]] = []
    repeat_penalty: Optional[float] = 1.1
    top_k: Optional[int] = 40
    streaming: bool = True

    # llm keyword arguments
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
    last_n_tokens_size: Optional[int] = 64
    use_mmap: Optional[bool] = True

    @root_validator()
    def init(cls, values: Dict) -> Dict:
        model_path = values["model_path"]
        model_kwargs = values["model_kwargs"]
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
        for k, v in values.items():
            if k in model_param_names:
                model_kwargs[k] = v

        if not values.get("n_gpu_layers", None):
            del values["n_gpu_layers"]

        try:
            values["model"] = Llama(model_path, **model_kwargs)

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
            result = self.model(prompt=prompt, **params)
            return result["choices"][0]["text"]

    def stream_(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Generator[Dict, None, None]:

        params = self._get_parameters(stop)
        result = self.model(prompt=prompt, stream=True, **params)
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
        return len(self.model.tokenize(text.encode("utf-8")))
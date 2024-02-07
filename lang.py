import json
import logging
import os
import sys
import time
from typing import Any, Union
from pathlib import Path

from bigdl.llm import llm_convert
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain, LLMMathChain, load_chain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.callbacks import CallbackManager
from langchain_core.language_models import LLM
from langchain_core.runnables import RunnableConfig
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer

from util.llm import AliceEmbedding, AliceLLM
from util.memory import ExtendedConversationBufferMemory


class AliceLLMAI:
    original_text_model_path: Union[str, Path] = "./original_text_model"
    original_vision_model_path: Union[str, Path] = "./original_vision_model"
    default_text_model_path: Union[str, Path] = r"./models/text_model"
    default_vision_model_path: Union[str, Path] = r"./models/vision_model"
    temp_path: Union[str, Path] = "./temp"
    data_file: Union[str, Path] = ""

    config_file_path: Union[str, Path] = "./config/config.json"

    human_prefix: str = "Human"
    ai_prefix: str = "AI"

    n_thread: int = 12

    prompt_template: str = (
        "[INST] <<SYS>>\n"
        "You are a helpful assistant. 你是一个乐于助人的助手。\n"
        "Today's date: {date}\n"
        "Time: {time}\n"
        "<</SYS>>\n\n"
        "{memory}\n"
        "{human_prefix}: {input}\n"
        "{ai_prefix}: [/INST]"
    )

    text_model_llm: LLM
    vision_model_llm: Any

    current_chain: Chain
    __conversation_chain: ConversationChain
    __math_chain: LLMMathChain

    def convert_text_model(self, model_path):
        """Transform model to native formate."""
        self.logger.info(f"Transforming the model to native format...\n{'=' * 40}")
        bigdl_llm_path = llm_convert(
            model=model_path,
            outfile=self.default_text_model_path,
            outtype='int4',
            tmp_path=self.temp_path,
            model_family="llama")
        self.logger.info(f"{'=' * 40}\nSuccessfully transformed the model to native format!")
        return bigdl_llm_path

    def convert_vision_model(self, model_path):
        ...

    def __init__(self,
                 text_model: Any = None,
                 vision_model: Any = None,
                 debug: bool = False,
                 **kwargs):
        # Initiate logger
        self.is_debug = debug
        self.logger = logging.getLogger("alice_ai")
        if debug:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Initiate models and embedding.
        text_model_path: str = ""
        if not text_model:
            if os.path.exists(self.config_file_path):
                _k = "text_model_path"
                try:
                    with open(self.config_file_path, mode="r", encoding="utf-8") as f:
                        text_model_path = json.loads(f.read())[_k]
                except KeyError:
                    self.logger.warning(
                        f"Fail to initiate model, because parameter'{_k}' was missed in the config file.")
                except json.decoder.JSONDecodeError:
                    self.logger.warning("Failed to load config file. Using default value.")
            elif os.path.exists(self.default_text_model_path):
                text_model_path = self.default_text_model_path
            elif os.path.exists(self.original_text_model_path):
                os.mkdir(self.default_text_model_path)
                text_model_path = self.convert_text_model(self.original_text_model_path)
                _to_write = {"text_model_path": text_model_path}
                with open(self.config_file_path, mode="w+") as f:
                    if content := f.read():
                        content = json.loads(content)
                        _to_write.update(content)
                    f.write(json.dumps(_to_write))

            if text_model_path:
                self.logger.info(f"Loading text model from '{text_model_path}'")
                self.text_model_llm, self.text_embedding = self.init_text_model(path=text_model_path)
            else:
                self.logger.error("Fail to load text model, because the path of the model is empty'.")
                sys.exit(1)

        # Initiate vectorstore
        self.text_vector_store = self.init_vector_database()

        # Initiate prompt
        input_variables = ["date", "current_time", "memory", "input"]
        try:
            with open("models/text_model/prompt.txt", encoding="utf-8", mode="r") as f:
                prompt_template = f.read()
                if not prompt_template:
                    raise FileExistsError
        except (FileNotFoundError, FileExistsError, UnicodeError):
            prompt_template = self.prompt_template
            with open("models/text_model/prompt.txt", encoding="utf-8", mode="w") as f:
                f.write(self.prompt_template)
        prompt_template = \
            (prompt_template.replace("{human_prefix}", kwargs.get("human_prefix", "Human"))
             .replace("{ai_prefix}", kwargs.get("ai_prefix", "AI")))
        self.prompt = PromptTemplate(template=prompt_template, input_variables=input_variables)

        # Initiate memory
        memory = ExtendedConversationBufferMemory(
            memory_key="memory",
            input_key="input",
            extra_variables=["date", "time"],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix
        )

        # Initiate chains
        self.current_chain = self.init_chain(memery=memory)

    def init_vector_database(self, embedding=None, data_file=None):
        embedding = embedding or self.text_embedding
        data_file = data_file or self.data_file

        loader = TextLoader(data_file)
        try:
            documents = loader.load()
        except FileNotFoundError as e:
            self.logger.error("Fail to initiate vectorstore.")
            raise e

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(texts, embedding)
        return vectorstore

    def init_text_model(self, path) -> tuple[LLM, Any]:
        model = AliceLLM(
            model_path=path,
            logger=self.logger,
            native=True,
            n_threads=self.n_thread,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=self.is_debug,
            n_ctx=2048,
            tempareture=0.8
        )
        tokenizer = AutoTokenizer.from_pretrained(self.original_text_model_path)
        embedding = AliceEmbedding(model=model, tokenizer=tokenizer)
        self.logger.info("Successfully loaded Alice LLM!")
        return model, embedding

    def init_vision_model(self, path):
        ...

    def init_chain(self, memery):
        self.__conversation_chain = ConversationChain(
            prompt=self.prompt,
            llm=self.text_model_llm,
            llm_kwargs={"max_new_tokens": 512},
            verbose=self.is_debug,
            memory=memery
        )
        self.__math_chain = LLMMathChain(
            prompt=self.prompt,
            llm_chain=self.__conversation_chain,
            verbose=self.is_debug
        )
        current_chain = self.__conversation_chain
        return current_chain

    def mode(self, mode: str = "chat") -> Any:
        mode = mode.lower()
        if mode == "chat":
            self.current_chain = self.__conversation_chain
        elif mode == "math":
            self.current_chain = self.__math_chain
        return self.current_chain

    def __del__(self):
        self.logger.info("Alice LLM shutdown.")

    def invoke(self, input_: dict[str, Any],
               config_: RunnableConfig | None = None,
               **kwargs: Any) -> dict[str, Any]:

        in_ = {"date": time.strftime("%Y/%m/%d"), "time": time.strftime("%H/%M/%S")}
        in_.update(input_)
        return self.current_chain.invoke(in_, config_, **kwargs)


if __name__ == "__main__":
    alice = AliceLLMAI(debug=True)
    alice.mode("chat")
    while True:
        try:
            query = input("\nQ:")
        except KeyboardInterrupt:
            break
        if not query:
            continue
        if query.startswith("/"):
            query = query.removeprefix("/").upper()
            match query:
                case "EXIT":
                    break
                case "MODE MATH":
                    chain = alice.mode("math")
                case "MODE CHAT":
                    chain = alice.mode("chat")
            continue
        result = alice.invoke({"input": query})

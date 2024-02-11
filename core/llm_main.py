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
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.callbacks import CallbackManager
from langchain_core.language_models import LLM
from langchain_core.runnables import RunnableConfig
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, LlamaTokenizer

from util.llm import AliceEmbedding, AliceLLM
from util.memory import ExtendedConversationBufferMemory
from util.chain import AliceConversationChain


class AliceLLMAI:
    original_text_model_path: Union[str, Path] = "../original_text_model"
    original_vision_model_path: Union[str, Path] = "./original_vision_model"
    default_text_model_path: Union[str, Path] = r"../models/text_model"
    default_vision_model_path: Union[str, Path] = r"./models/vision_model"
    temp_path: Union[str, Path] = "../temp"
    data_file: Union[str, Path] = ""

    config_file_path: Union[str, Path] = "./config/config.json"

    human_prefix: str = "Human"
    ai_prefix: str = "Alice"

    n_thread: int = 12

    prompt_template: str = (
        "[INST] <<SYS>>\n"
        "Your are a helpful android. 你是一个乐于助人的仿生人。\n"
        "Your are an android named 'Tendon Arisu'. Your need to meet others' request as best as you can. \n"
        "When ask your name, you must respond with 'Tendon Alice' or 'Alice'. \n"
        "You now today's date.\n"
        "<</SYS>>\n\n"
        "{human_prefix}: 你叫什么名字 \n"
        "{ai_prefix}: 我的名字是Tendon Alice。请问有什么Alice可以帮助您的吗？ \n"
        "{human_prefix}: 今天是{date}。 \n"
        "{ai_prefix}: 今天是{date}。 \n"
        "{memory}\n"
        "{human_prefix}: {input}\n"
        "{ai_prefix}: [/INST]"
    )

    text_model_llm: LLM
    vision_model_llm: Any

    int_l: str = "8"

    current_chain: Chain
    __conversation_chain: ConversationChain
    __math_chain: LLMMathChain

    def convert_text_model(self, model_path):
        """Transform model to native formate."""
        self.logger.info(f"Transforming the model to native format...\n{'=' * 40}")
        bigdl_llm_path = llm_convert(
            model=model_path,
            outfile=self.default_text_model_path,
            outtype=f'int{self.int_l}',
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
        self.text_model_llm, self.text_embedding = self.init_text_model(path=text_model)

        # Initiate vectorstore
        self.text_vector_store = self.init_vector_database()

        # Initiate prompt
        input_variables = ["memory", "input", "date"]
        try:
            with open("../models/text_model/prompt.txt", encoding="utf-8", mode="r") as f:
                prompt_template = f.read()
                if not prompt_template:
                    raise FileExistsError
        except (FileNotFoundError, FileExistsError, UnicodeError):
            prompt_template = self.prompt_template
            with open("../models/text_model/prompt.txt", encoding="utf-8", mode="w") as f:
                f.write(self.prompt_template)
        prompt_template = \
            (prompt_template.replace("{human_prefix}", kwargs.get("human_prefix", "Human"))
             .replace("{ai_prefix}", kwargs.get("ai_prefix", "AI")))
        self.prompt = PromptTemplate(template=prompt_template, input_variables=input_variables)

        # Initiate memory
        memory = ExtendedConversationBufferMemory(
            k=10,
            memory_key="memory",
            input_key="input",
            extra_variables=['date'],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
            summarize_step=2
        )

        # Initiate chains
        self.current_chain = self.init_chain(memery=memory)

    def init_vector_database(self, embedding=None, data_file=None):
        embedding = embedding or self.text_embedding
        data_file = data_file or self.data_file

        loader = TextLoader(data_file)
        try:
            documents = loader.load()
        except RuntimeError as e:
            self.logger.error(f"Fail to initiate vectorstore.({e})")
            return None

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(texts, embedding)
        return vectorstore

    def init_text_model(self, path, _jump: int = 0) -> tuple[LLM, Any]:
        def init_model(path_):
            self.logger.info(f"Loading text model from '{path}'")
            model = AliceLLM(
                model_path=path_,
                logger=self.logger,
                model_kwargs=dict(
                    n_threads=self.n_thread,
                    n_ctx=2048),
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                verbose=self.is_debug,
                tempareture=0.8
            )
            tokenizer = AutoTokenizer.from_pretrained(self.default_text_model_path)
            embedding = AliceEmbedding(model=model, tokenizer=tokenizer)
            self.logger.info("Successfully loaded Alice LLM!")
            return model, embedding

        if path and _jump != 1:
            try:
                text_model_llm, text_embedding = init_model(path)
            except Exception as e:
                self.logger.warning(
                    f"""Failed to initiate text model, because 'path' argument was invalid.
                     Retry to use other ways to initiate model.({e})""")
                text_model_llm, text_embedding = self.init_text_model(None, 1)

        else:
            if os.path.exists(self.config_file_path) and _jump != 2:
                _k = "text_model_path"
                try:
                    config_file = Path(self.config_file_path)
                    if content := config_file.read_text():
                        path = json.loads(content)[_k]
                    else:
                        path = self.default_text_model_path
                        raise json.decoder.JSONDecodeError
                except KeyError:
                    self.logger.warning(
                        f"Failed to initiate model, because parameter'{_k}' was missed in the config file.")
                except json.decoder.JSONDecodeError:
                    self.logger.warning("Failed to load config file. Using default value.")
                text_model_llm, text_embedding = init_model(path + f"/bigdl_llm_llama_q{self.int_l}_0.bin")

            elif os.path.exists(self.default_text_model_path) and _jump != 3:
                try:
                    text_model_llm, text_embedding = init_model(
                        self.default_text_model_path + f"/bigdl_llm_llama_q{self.int_l}_0.bin")
                except Exception as e:
                    self.logger.warning(f"""Failed to initiate text models from default path but original models is exist.
                    Retrying to load from original models...({e})""")
                    text_model_llm, text_embedding = self.init_text_model(path, 3)

            elif os.path.exists(self.original_text_model_path) and _jump != 4:
                if not os.path.exists(self.default_text_model_path):
                    os.mkdir(self.default_text_model_path)
                _tokenizer = LlamaTokenizer.from_pretrained(self.original_text_model_path, trust_remote_code=True)
                _tokenizer.save_pretrained(self.default_text_model_path)
                path = self.convert_text_model(self.original_text_model_path)
                _to_write = {"text_model_path": path}
                config_file = Path(self.config_file_path)
                if config_file.exists():
                    content = json.loads(config_file.read_text())
                    _to_write.update(content)
                    config_file.write_text(json.dumps(_to_write))
                text_model_llm, text_embedding = init_model(path)
                self.logger.info("Deleting temp file...")
                try:
                    os.remove(self.temp_path)
                except:
                    pass
            else:
                self.logger.error("Failed to load model.")
                raise ValueError

        return text_model_llm, text_embedding

    def init_vision_model(self, path):
        ...

    def init_chain(self, memery):
        self.__conversation_chain = AliceConversationChain(
            prompt=self.prompt,
            llm=self.text_model_llm,
            llm_kwargs={"max_new_tokens": 512},
            verbose=self.is_debug,
            memory=memery
        )
        self.__conversation_chain.save_path = "./data/memory/leaf.json"
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
               complete: bool = False,
               **kwargs: Any) -> dict[str, Any]:

        in_ = {"date": time.strftime("%Y年%m月%d日")}
        input_.update(in_)
        if complete:
            return self.current_chain.invoke(input_, config_, **kwargs)
        else:
            return self.current_chain.invoke(input_, config_, **kwargs)["response"]


def example():
    alice = AliceLLMAI(debug=False)
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
        # print(result)

if __name__ == "__main__":
    example()

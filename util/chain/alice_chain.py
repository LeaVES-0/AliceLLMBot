from pathlib import Path
from typing import Union, Any

from langchain.chains import ConversationChain


class AliceConversationChain(ConversationChain):
    save_path: str = "./data/memory/"

    def __del__(self):
        self.save(self.save_path)

from pathlib import Path
from typing import Union

from langchain.chains import ConversationChain


class AliceConversationChain(ConversationChain):
    save_path: Union[Path, str]

    def __del__(self):
        self.save(self.save_path)

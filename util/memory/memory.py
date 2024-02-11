from pathlib import Path
from typing import List, Dict, Any, Union

from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import get_buffer_string, BaseMessage


class ExtendedConversationBufferMemory(ConversationBufferWindowMemory):

    extra_variables: List = None

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables."""
        return [self.memory_key] + self.extra_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return buffer with history and extra variables"""
        d = super().load_memory_variables(inputs)
        d.update({k: inputs.get(k) for k in self.extra_variables})
        return d


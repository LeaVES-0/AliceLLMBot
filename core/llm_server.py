import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict
from functools import cache
from llm_inference import AliceLLMAI
from util.api_server.api_server import AliceApiServerBase


class AliceLLMServer(AliceApiServerBase):
    def __init__(self):
        super().__init__()
        self.ai_core = AliceLLMAI(debug=False)

    async def get_api_message(self, user_input: Dict):
        loop = asyncio.get_event_loop()
        pool_executor = ThreadPoolExecutor()
        future = await loop.run_in_executor(pool_executor, lambda: self.ai_core.invoke(user_input))
        return future

    @cache
    def api_key(self) -> List:
        f = Path("./data/tokens.txt")
        token_list = f.read_text().splitlines()
        return token_list


if __name__ == "__main__":
    server = AliceLLMServer()
    server.run()

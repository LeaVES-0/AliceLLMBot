import json
import time
from abc import ABC, abstractmethod
from typing import Dict

from aiohttp import web
from aiohttp.web_request import Request
from aiohttp.web_response import Response


class AliceApiServerBase(ABC):
    response_body = {
        "api_version": 1.0,
        "output": dict(),
        "trace_id": "",
        "timestamp": float(),
        "error": ""
    }

    @abstractmethod
    def api_key(self):
        ...

    def __init__(self, address=('127.0.0.1', 18080)):
        self.app = web.Application()
        self.app.add_routes(
            [
                web.get("/", self.http_handle),  # curl http://localhost:8080/
                web.get("/{key}", self.http_handle),  # curl http://localhost:8080/
            ]
        )
        self.address = address

    @abstractmethod
    def get_api_message(self, user_input) -> Dict:
        ...

    async def http_handle(self, request: Request | None = None) -> Response:
        """
        HTTP 请求处理器
        """
        # web.get("/{name}", ...) 中的 name
        key = request.match_info.get("key")
        if not key:
            self.response_body['error'] = "Please provide an apikey. (Format: [HOST]/[API_KEY])"
        elif key != self.api_key:
            self.response_body['error'] = "Apikey was incorrect."
        else:
            try:
                data = json.loads(await request.content.read())
                self.response_body['output'] = self.get_api_message(data["input"])
                self.response_body['trace_id'] = str(data["trace_id"])
            except json.JSONDecodeError:
                self.response_body['error'] = f"The format of the request was incorrect."
            except KeyError:
                self.response_body['error'] = "A necessary argument was missed."

        self.response_body["timestamp"] = time.time()
        resp_text = json.dumps(self.response_body, indent=4)

        # 返回一个 响应对象
        return web.Response(status=200,
                            text=resp_text,
                            content_type="text/plain")

    def run(self):
        host, port = self.address
        web.run_app(self.app, host=host, port=port)


class AliceApiServerTest(AliceApiServerBase):
    @property
    def api_key(self):
        return "MillenniumAlice25565"

    def get_api_message(self, user_input) -> Dict:
        return {"output": "test"}


def test():
    test_instance = AliceApiServerTest()
    test_instance.run()


if __name__ == "__main__":
    test()

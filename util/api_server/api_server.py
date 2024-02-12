import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List

from aiohttp import web
from aiohttp.web_request import Request
from aiohttp.web_response import Response


class AliceApiServerBase(ABC):
    limit: int = 1

    @property
    def response_body(self):
        response_body = {
            "api_version": 1.0,
            "output": dict(),
            "trace_id": "",
            "timestamp": float(),
            "info": {"code": int(),
                     "msg": ""}
        }
        return response_body

    @abstractmethod
    def api_key(self) -> List:
        ...

    def __init__(self, address=('127.0.0.1', 18080)):
        self.n_process = 0
        self.app = web.Application()
        self.app.add_routes(
            [
                web.get("/", self.http_handle),  # curl http://localhost:8080/
                web.get("/{key}", self.http_handle),  # curl http://localhost:8080/
                web.post("/", self.http_handle),  # curl http://localhost:8080/
                web.post("/{key}", self.http_handle),  # curl http://localhost:8080/
            ]
        )
        self.address = address

    @abstractmethod
    async def get_api_message(self, user_input: Dict) -> Dict:
        pass

    async def http_handle(self, request: Request | None = None) -> Response:
        """
        HTTP 请求处理器
        """
        key = request.match_info.get("key", None)
        if not key:
            return self.set_response(40)
        elif key not in self.api_key():
            return self.set_response(41)
        try:
            data = json.loads(await request.content.read())
        except json.JSONDecodeError:
            return self.set_response(42)

        if self.n_process >= self.limit:
            return self.set_response(20)

        try:
            match data:
                case {"input": input_dict, "trace_id": trace_id, **kw}:
                    self.n_process += 1
                    result = await self.get_api_message(input_dict)
                    self.n_process -= 1
                    return self.set_response(0, output=result, trace_id=trace_id)
                case _:
                    return self.set_response(42)
        except Exception as e:
            return self.set_response(50, msg=e)

    def set_response(self, code, transform: bool = True, **kwargs) -> Dict | web.Response:
        response = self.response_body
        msg_map = {
            0: "Successful!",
            40: "Please provide an apikey. (Format: [HOST]/[API_KEY])",
            41: "Apikey was incorrect.",
            42: "The format of the request was incorrect.",
            50: "An unexpected error occurred, please try again.",
            20: "Server is busy at this time."
        }
        if code not in msg_map.keys():
            raise ValueError
        msg = msg_map[code]
        response["info"]['msg'] = msg
        response["info"]["code"] = code
        response["timestamp"] = time.time()
        if code != 0:
            del response["output"]
        else:
            for k, v in kwargs.items():
                if k == "msg":
                    response["info"]['msg'] += str(v)
                    continue
                response[k] = v
        if transform:
            return web.Response(status=200, text=json.dumps(response, indent=4), content_type="text/plain")
        else:
            return response

    def run(self, loop=None):
        host, port = self.address
        web.run_app(self.app, host=host, port=port, loop=loop)


class AliceApiServerTest(AliceApiServerBase):
    @property
    def api_key(self) -> List:
        return [
            "MillenniumAlice25565",
        ]

    def get_api_message(self, user_input) -> Dict:
        return {"output": "test"}


def test():
    test_instance = AliceApiServerTest()
    test_instance.run()


if __name__ == "__main__":
    test()

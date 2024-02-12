import json
import pprint

import requests
import uuid

kw = {
    'trace_id': uuid.uuid4().hex,
    'input': {'input': '你好'}
}

response = requests.post("http://127.0.0.1:18080/eQfbYGi0hi9pO83HU6Ii", json=kw)

print("state_code:", response.status_code)
pprint.pprint(json.loads(response.text))

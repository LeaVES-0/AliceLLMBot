import json
import pprint

import requests
import uuid

def ask(input):
    kw = {
        'trace_id': uuid.uuid4().hex,
        'input': {'input': input}
    }

    response = requests.post("http://millennium.ink:18080/eQfbYGi0hi9pO83HU6Ii", json=kw)

    print("state_code:", response.status_code)
    pprint.pprint(json.loads(response.text))

while True:
    ask_q = input("Q:")
    ask(ask_q)
    
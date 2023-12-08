import openai
from openai import OpenAI
import signal
import tqdm.notebook as tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from os.path import expanduser

from environments.hallway_env import prompt

import base64
import requests

from utils.vlm_utils import timeout, encode_image, response_pre_check

# Set OpenAI API key.
openai_api_key = "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"  # Justin's key
openai.api_key = openai_api_key


def make_payload(prompt, max_tokens, seed, image_path):
    base64_image = encode_image(image_path)
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        # "temperature": 0,
        # "seed": seed
    }
    return payload

# OpenAI only supports up to five tokens (logprobs argument) for getting the likelihood.
# Thus we use the logit_bias argument to force LLM only consdering the five option
# tokens: A, B, C, D, E
def vlm(prompt,
        image_path,
       max_tokens=300,
       temperature=0,
       seed=1234,
       timeout_seconds=30,
       max_attempts=5):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    payload = make_payload(prompt, max_tokens, seed, image_path)
    for _ in range(max_attempts):
        try:
            with timeout(seconds=timeout_seconds):

                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                if not response_pre_check(response):
                    raise ValueError(f"Probability list had unexpected shape. Check message.")
            break
        except:
            print('Timeout, retrying...')
            pass
    return response

def hallway_parse_response(response):
    response_str = response.json()["choices"][0]["message"]["content"]
    print(response_str)
    probs = re.findall(r"[-+]?(?:\d*\.*\d+)%", response_str)
    probs = [float(x.split('%')[0]) for x in probs]
    return np.array(probs)

if __name__ == "__main__":
    home = expanduser("~")
    image_path = os.path.join(home, 'PredictiveRL/try.png')

    response = vlm(prompt, image_path, max_tokens=300, seed=1234)
    probs = hallway_parse_response(response)
    print(probs)
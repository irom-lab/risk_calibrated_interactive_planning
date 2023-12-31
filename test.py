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

# from environments.hallway_env import prompt

import base64
import requests

from utils.vlm_utils import timeout, encode_image, response_pre_check

# Set OpenAI API key.
# openai_api_key = "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"  # Justin's key
openai_api_key = "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"  # Ariel's key
openai.api_key = openai_api_key

prompt = "Write me a poem."


def make_payload(prompt, max_tokens, seed, image_path, is_dir=True):
    if is_dir:
        image_paths = sorted(os.listdir(image_path))
        image_paths = [os.path.join(image_path, p) for p in image_paths]
    else:
        image_paths  = [image_path]
    base64_images = [encode_image(path) for path in image_paths]
    content_message = {"type": "text", "text": prompt}

    def image_prompt(base64_image):
        ret = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        return ret

    content = [content_message]
    for i in base64_images:
        content.append(image_prompt(i))

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": seed
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
       max_attempts=10,
       intent_set_size=12,
       is_dir=True):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    payload = make_payload(prompt, max_tokens, seed, image_path, is_dir)
    for _ in range(max_attempts):
        try:
            with timeout(seconds=timeout_seconds):

                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            

                if not response_pre_check(response, intent_set_size):
                    raise ValueError(f"Probability list had unexpected shape. Check message.")
            break
        except:
            print('Invalid response, retrying...')
            pass

    return response

def parse_response(response):
    response_str = response.json()["choices"][0]["message"]["content"]
    print(response_str)
    probs = re.findall(r"[-+]?(?:\d*\.*\d+)%", response_str)
    probs = [float(x.split(':')[0]) for x in probs]
    return np.array(probs)

if __name__ == "__main__":
    home = expanduser("~")
    # image_path = os.path.join(home, 'PredictiveRL/franka_img_test/blocks.png')

    # Windows
    # image_path = os.path.join(home, 'Downloads/Princeton/F2023/SeniorThesis/PredictiveRL/franka_img_test/blocks.png')

    # Linux
    image_path = "/mnt/c/Users/Ariel/Downloads/Princeton/F2023/SeniorThesis/PredictiveRL/franka_img_test/blocks.png"
    # qa, alpha_ids = gen_prompt_qa()
    # qa_str = '\n'.join(qa)
    # prompt_full = prompt_pre + "\n" + qa_str + " \n" + prompt_suffix
    # print(prompt_full)
    prompt_full = "Write me a poem please."

    response = vlm(prompt_full, image_path, max_tokens=300, seed=1234, is_dir=False, intent_set_size=12)
    probs = parse_response(response)
    print(probs)
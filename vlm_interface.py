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

from utils.vlm_utils import timeout, encode_image, response_pre_check, extract_probs

# Set OpenAI API key.
# openai_api_key = "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"  # Justin's key
openai_api_key = "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"  # Ariel's key
openai.api_key = openai_api_key

prob_bins = list(np.arange(0, 101, 10))

prompt_pre = f"Here is an image of a collection of wooden blocks, sorted by: shape, color, or size. \n " \
    + f"For each labeled group, choose an approximate numerical probability {prob_bins} for each sorting method: "

prompt_suffix = "Give your response as a list organized by each letter. If shapes, sizes, and colors are only slightly different, treat them as identical. Please return the list in the following format: \n [probability for method (A)] \n [probability for method (B)] \n [probability for method (C)] \n etc. Please print the list and only the list."
# prompt_suffix = "Give your response as a list organized by each letter. If shapes, sizes, and colors are only slightly different, treat them as identical. "
prompt_suffix = "Give your response as a list separated by newlines."

def next_alpha(s):
    return chr((ord(s.upper())+1 - 65) % 26 + 65)

def gen_prompt_qa(num_groups=4, sorting_types=["shape", "color", "size"]):

    strs = []
    alpha_ids = []
    alpha = "A"
    for g in range(num_groups):
        sg = str(g+1)
        for st in sorting_types:
            new_s = f"({alpha}): Group {sg} by {st}"
            strs.append(new_s)
            alpha = next_alpha(alpha)
            alpha_ids.append(sg)
    return strs, alpha_ids




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
    print("parse response response string")
    print(response_str)
    probs = re.findall(r"[-+]?(?:\d*\.*\d+)%", response_str)
    print("parse response probs")
    print(probs)
    probs = [float(x.split(':')[0]) for x in probs]
    return np.array(probs)

if __name__ == "__main__":
    home = expanduser("~")
    # image_path = os.path.join(home, 'PredictiveRL/franka_img_test/blocks.png')

    # Windows
    # image_path = os.path.join(home, 'Downloads/Princeton/F2023/SeniorThesis/PredictiveRL/franka_img_test/blocks.png')

    # Linux
    image_path = "/mnt/c/Users/Ariel/Downloads/Princeton/F2023/SeniorThesis/PredictiveRL/franka_img_test/blocks_with_circles.png"
    qa, alpha_ids = gen_prompt_qa()
    qa_str = '\n'.join(qa)
    prompt_full = prompt_pre + "\n" + qa_str + " \n" + prompt_suffix
    print(prompt_full)

    response = vlm(prompt_full, image_path, max_tokens=300, seed=1234, is_dir=False, intent_set_size=12)
    # probs = parse_response(response) # prints response.json()["choices"][0]["message"]["content"]
    probs = extract_probs(response)
    print("probs")
    print(probs) # prints array of probabilities
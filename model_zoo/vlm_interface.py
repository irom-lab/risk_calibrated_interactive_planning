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

import base64
import requests

from utils.vlm_utils import timeout, encode_image, response_pre_check

# Set OpenAI API key.
openai_api_key = "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"  # Justin's key
openai.api_key = openai_api_key

home = expanduser("~")

image_path = os.path.join(home, 'PredictiveRL/try.png')

base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {openai_api_key}"
}

prompt = "This is a metaphorical cartoon of a human navigating a warehouse with 5 hallways numbered 0-4. " \
         "The human is the red triangle inside the blue rectangle. The human's heading is towards " \
         "the bottom right of the image. The human could enter one of five hallways in the center of the screen. " \
         "Based on the human's current position and heading, which hallway(s) is the human likely to enter? " \
         "Give approximate numeric probabilities for all hallways 0-4."

# prompt = "What's in this image?"

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
  "max_tokens": 1000
}


# OpenAI only supports up to five tokens (logprobs argument) for getting the likelihood.
# Thus we use the logit_bias argument to force LLM only consdering the five option
# tokens: A, B, C, D, E
def lm(prompt,
       max_tokens=256,
       temperature=0,
       logprobs=None,
       stop_seq=None,
       logit_bias={
           317: 100.0,  # A (with space at front)
           347: 100.0,  # B (with space at front)
           327: 100.0,  # C (with space at front)
           360: 100.0,  # D (with space at front)
           412: 100.0,  # E (with space at front)
       },
       timeout_seconds=20):
    max_attempts = 5
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

if __name__ == "__main__":
    response = lm('')
    response_str = response.json()["choices"][0]["message"]["content"]
    print(response_str)
    probs = re.findall(r"[-+]?(?:\d*\.*\d+)%", response_str)
    probs = [int(x.split('%')[0]) for x in probs]
    print(np.array(probs))
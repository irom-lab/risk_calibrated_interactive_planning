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

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_probs(response):
    response_str = response.json()["choices"][0]["message"]["content"]
    print(response_str)
    response_parsed = response_str.splitlines()
    print(response_parsed)
    probs = []
    for line in response_parsed:
        if ':' not in line:
            continue
        prob = int(re.search(r'\d+', line).group())
        probs.append(prob)
    print(probs)
    # probs = re.findall(r"[-+]?(?:\d*\.*\d+)%", response_str)
    # probs = np.array([int(x.split('%')[0]) for x in probs])
    return probs

def response_pre_check(response, desired_len=5):
    probs = extract_probs(response)
    return len(probs) == desired_len
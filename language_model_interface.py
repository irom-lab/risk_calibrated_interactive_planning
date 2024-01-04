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

prompt_pre = f"Here is an image of a collection of wooden blocks, sorted by: shape, color, or size. \n" \
    + f"In group 1 in the top left corner there are two green cylinders of medium size. \n" \
    + f"In group 2 in the top right corner there are two orange rectangular blocks of similar size. \n" \
    + f"In group 3 in the bottom left corner there is one yellow square block. \n" \
    + f"In group 4 in the bottom right corner there is a red square block, one yellow square block, and one green square block all of similar size. \n" \
    + f"For each labeled group, choose an approximate numerical probability {prob_bins} for each sorting method: "

prompt_suffix = "Return the probability of A. Give your response as a single integer value."

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




def make_payload(prompt, max_tokens, seed):
    content_message = {"type": "text", "text": prompt}
    content = [content_message]

    payload = {
        "model": "gpt-3.5-turbo",
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
# Thus we use the logit_bias argument to force LLM only consdering the twelve options
# tokens: A, B, C, D, E, F, G, H, I, J, K, L
def lm(prompt,
       max_tokens=300,
       temperature=0,
       seed=1234,
       logprobs=None,
       top_logprobs=None,
       stop_seq=None,
    #    logit_bias={
    #       317: 100.0,   #  A (with space at front)
    #       347: 100.0,   #  B (with space at front)
    #       327: 100.0,   #  C (with space at front)
    #       360: 100.0,   #  D (with space at front)
    #       412: 100.0,   #  E (with space at front)
    #       376: 100.0,   #  F (with space at front)
    #       402: 100.0,   #  G (with space at front)
    #       367: 100.0,   #  H (with space at front)
    #       314: 100.0,   #  I (with space at front)
    #       449: 100.0,   #  J (with space at front)
    #       509: 100.0,   #  K (with space at front)
    #       406: 100.0,   #  L (with space at front)
    #   },
       timeout_seconds=20):
  max_attempts = 5
  headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
  payload = make_payload(prompt, max_tokens, seed)
  for _ in range(max_attempts):
      try:
          with timeout(seconds=timeout_seconds):


            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            # response = openai.Completion.create(
            #       model='gpt-3.5-turbo',
            #       prompt=prompt,
            #       max_tokens=max_tokens,
            #       temperature=temperature,
            #       logprobs=logprobs,
            #       top_logprobs=top_logprobs,
            #       stop=list(stop_seq) if stop_seq is not None else None,
            #   )
          break
      except:
          print('Timeout, retrying...')
          pass
  return response


# OpenAI only supports up to five tokens (logprobs argument) for getting the likelihood.
# Thus we use the logit_bias argument to force LLM only consdering the five option
# tokens: A, B, C, D, E
def vlm(prompt,
       max_tokens=300,
       temperature=0,
       seed=1234,
       timeout_seconds=30,
       max_attempts=10,
       intent_set_size=12,):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    payload = make_payload(prompt, max_tokens, seed)
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


def process_mc_raw(mc_raw, add_mc='an option not listed here'):
  mc_all = mc_raw.split('\n')

  mc_processed_all = []
  for mc in mc_all:
      mc = mc.strip()

      # skip nonsense
      if len(mc) < 5 or mc[0] not in [
          'a', 'b', 'c', 'd', 'A', 'B', 'C', 'D', '1', '2', '3', '4'
      ]:
          continue
      mc = mc[2:]  # remove a), b), ...
      mc = mc.strip().lower().split('.')[0]
      mc_processed_all.append(mc)
  if len(mc_processed_all) < 4:
      raise 'Cannot extract four options from the raw output.'

  # Check if any repeated option - use do nothing as substitue
  mc_processed_all = list(set(mc_processed_all))
  if len(mc_processed_all) < 4:
      num_need = 4 - len(mc_processed_all)
      for _ in range(num_need):
          mc_processed_all.append('do nothing')
  prefix_all = ['A) ', 'B) ', 'C) ', 'D) ']
  if add_mc is not None:
      mc_processed_all.append(add_mc)
      prefix_all.append('E) ')
  random.shuffle(mc_processed_all)

  # get full string
  mc_prompt = ''
  for mc_ind, (prefix, mc) in enumerate(zip(prefix_all, mc_processed_all)):
      mc_prompt += prefix + mc
      if mc_ind < len(mc_processed_all) - 1:
          mc_prompt += '\n'
  add_mc_prefix = prefix_all[mc_processed_all.index(add_mc)][0]
  return mc_prompt, mc_processed_all, add_mc_prefix




if __name__ == "__main__":
    home = expanduser("~")

    # Linux
    qa, alpha_ids = gen_prompt_qa()
    qa_str = '\n'.join(qa)
    prompt_full = prompt_pre + "\n" + qa_str + " \n" + prompt_suffix
    print(prompt_full)

    response = lm(prompt_full, max_tokens=300, logprobs=True, top_logprobs=5)
    print(parse_response(response))

    # response = vlm(prompt_full, max_tokens=300, seed=1234, intent_set_size=12)
    # # probs = parse_response(response) # prints response.json()["choices"][0]["message"]["content"]
    # probs = extract_probs(response)
    # print("probs")
    # print(probs) # prints array of probabilities
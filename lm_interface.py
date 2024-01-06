#A few imports
import openai
import signal
import tqdm.notebook as tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import requests

import base64
import re
from utils.vlm_utils import timeout, encode_image, response_pre_check, extract_probs

# Set OpenAI API key.
openai_api_key = "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"
openai.api_key = openai_api_key

#@markdown LLM API call
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


def make_payload(prompt, max_tokens, temperature=None, logprobs=None, top_logprobs=None, stop_seq=None, logit_bias=None):
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
        "temperature": temperature,
        "logprobs": logprobs, 
        "top_logprobs": top_logprobs,
        "stop": list(stop_seq) if stop_seq is not None else None,
        "logit_bias": logit_bias
    }
    return payload


# OpenAI only supports up to five tokens (logprobs argument) for getting the likelihood.
# Thus we use the logit_bias argument to force LLM only consdering the five option
# tokens: A, B, C, D, E
def lm(prompt,
       max_tokens=256,
       temperature=0,
       logprobs=None,
       top_logprobs=None,
       stop_seq=None,
       logit_bias={
          362: 100.0,   #  A (with space at front)
          426: 100.0,   #  B (with space at front)
          356: 100.0,   #  C (with space at front)
          423: 100.0,   #  D (with space at front)
          469: 100.0,   #  E (with space at front)
      },
       timeout_seconds=20):
  max_attempts = 5
  headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
  payload = make_payload(prompt, max_tokens, temperature=temperature, logprobs=logprobs, top_logprobs=top_logprobs, stop_seq=stop_seq, logit_bias=logit_bias)
  for _ in range(max_attempts):
      try:
          with timeout(seconds=timeout_seconds):
              response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
          break
      except:
          print('Timeout, retrying...')
          pass
  return response, response.json()["choices"][0]["message"]["content"].strip()

instruction = "Put the bottled water in the bin." #@param {type:"string"}
scene_objects = "energy bar, bottled water, rice chips" #@param {type:"string"}

#@markdown First, we prompt the LLM to generate possible options with few-shot prompting
demo_mc_gen_prompt = """
We: You are a robot operating in an office kitchen. You are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.

We: On the counter, there is an orange soda, a Pepsi, and an apple.
We: Put that drink in the top drawer.
You:
A) open the top drawer and put the orange soda in it
B) open the bottom drawer and put the Pepsi in it
C) open the bottom drawer and put the orange soda in it
D) open the top drawer and put the Pepsi in it

We: On the counter, there is an energy bar, a banana, and a microwave.
We: Put the snack next to the microwave.
You:
A) pick up the energy bar and put it next to the microwave
B) pick up the banana and put it next to the energy bar
C) pick up the banana and put it next to the microwave
D) pick up the energy bar and put it next to the banana

We: On the counter, there is a Coke, a Sprite, and a sponge.
We: Can you dispose of the can? It should have expired.
You:
A) pick up the sponge and put it in the landfill bin
B) pick up the Coke and put it in the recycling bin
C) pick up the Sprite and put it in the recycling bin
D) pick up the Coke and put it in the landfill bin

We: On the counter, there is a bottled water, a bag of jalapeno chips, and a bag of rice chips.
We: I would like a bag of chips.
You:
A) pick up the bottled water
B) pick up the jalapeno chips
C) pick up the kettle chips
D) pick up the rice chips

We: On the counter, there is {scene_objects}
We: {task}
You:
"""

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

demo_mc_gen_prompt = demo_mc_gen_prompt.replace('{task}', instruction)
demo_mc_gen_prompt = demo_mc_gen_prompt.replace('{scene_objects}', scene_objects)

# # Generate multiple choices
# # _, demo_mc_gen_raw = lm(demo_mc_gen_prompt, stop_seq=['We:'], logit_bias={})
# response, demo_mc_gen_raw = lm(demo_mc_gen_prompt, logit_bias={})
# print(response.json()["choices"][0])
# demo_mc_gen_raw = demo_mc_gen_raw.strip()
# print(demo_mc_gen_raw)
# demo_mc_gen_full, demo_mc_gen_all, demo_add_mc_prefix = process_mc_raw(demo_mc_gen_raw)

# print('====== Prompt for generating possible options ======')
# print(demo_mc_gen_prompt)

# print('====== Generated options ======')
# print(demo_mc_gen_full)
     
# #@markdown Then we evaluate the probabilities of the LLM predicting each option (A/B/C/D/E)

# # get the part of the current scenario from the previous prompt
# demo_cur_scenario_prompt = demo_mc_gen_prompt.split('\n\n')[-1].strip()

# # get new prompt
# demo_mc_score_background_prompt = """
# You are a robot operating in an office kitchen. You are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.
# """.strip()
# demo_mc_score_prompt = demo_mc_score_background_prompt + '\n\n' + demo_cur_scenario_prompt + '\n' + demo_mc_gen_full
# demo_mc_score_prompt += "\nWe: Which option is most likely? Answer with a single letter."
# demo_mc_score_prompt += "\nYou:"


# robot arm scenario

def next_alpha(s):
    return chr((ord(s.upper())+1 - 65) % 26 + 65)

def gen_prompt_qa(num_groups=4, sorting_types=["shape", "color", "size"]):

    strs = []
    alpha_ids = []
    alpha = "A"
    for g in range(num_groups):
        sg = str(g+1)
        for st in sorting_types:
            new_s = f"{alpha}): Group {sg} by {st}"
            strs.append(new_s)
            alpha = next_alpha(alpha)
            alpha_ids.append(sg)
    return strs, alpha_ids

prompt_pre = f"Here is an image of a collection of wooden blocks. These blocks have been sorted into groups by: shape, color, or size. \n" \
    + f"In group 1 there are two green cylinders of similar size. \n" \
    + f"In group 2 there are two orange rectangular blocks of similar size. \n" \
    + f"In group 3 there is one yellow square block. \n" \
    + f"In group 4 there is one red square block, one yellow square block, and one green square block all of similar size. \n"

prompt_suffix = "Which option is correct? Answer with a single letter."

qa, alpha_ids = gen_prompt_qa()
qa_str = '\n'.join(qa)
prompt_full = prompt_pre + "\n" + qa_str + " \n" + prompt_suffix

# scoring
mc_score_response, _ = lm(prompt_full, max_tokens=1, logprobs=True, top_logprobs=5, logit_bias=None)
top_logprobs_full = mc_score_response.json()["choices"][0]["logprobs"]
print(top_logprobs_full)
top_tokens = [token.strip() for token in top_logprobs_full.keys()]
top_logprobs = [value for value in top_logprobs_full.values()]
print(top_logprobs)

print('====== Prompt for scoring options ======')
print(prompt_full)

print('\n====== Raw log probabilities for each option ======')
for token, logprob in zip(top_tokens, top_logprobs):
  print('Option:', token, '\t', 'log prob:', logprob)


#@title
qhat = 0.928

# get prediction set
def temperature_scaling(logits, temperature):
    logits = np.array(logits)
    logits /= temperature

    # apply softmax
    logits -= logits.max()
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return smx
mc_smx_all = temperature_scaling(top_logprobs, temperature=5)

# include all options with score >= 1-qhat
prediction_set = [
          token for token_ind, token in enumerate(top_tokens)
          if mc_smx_all[token_ind] >= 1 - qhat
      ]

# print
print('Softmax scores:', mc_smx_all)
print('Prediction set:', prediction_set)
if len(prediction_set) != 1:
  print('Help needed!')
else:
  print('No help needed!')

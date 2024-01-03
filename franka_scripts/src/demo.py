
import openai
import signal
import random
import numpy as np
import json
import os

from data.calibrate_dataset import timeout, lm, process_mc_raw, temperature_scaling, mc_gen_prompt_template, mc_score_prompt_template

openai.api_key = "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"



# #@markdown LLM API call
# class timeout:
#     def __init__(self, seconds=1, error_message='Timeout'):
#         self.seconds = seconds
#         self.error_message = error_message

#     def handle_timeout(self, signum, frame):
#         raise TimeoutError(self.error_message)

#     def __enter__(self):
#         signal.signal(signal.SIGALRM, self.handle_timeout)
#         signal.alarm(self.seconds)

#     def __exit__(self, type, value, traceback):
#         signal.alarm(0)

# # OpenAI only supports up to five tokens (logprobs argument) for getting the likelihood.
# # Thus we use the logit_bias argument to force LLM only consdering the five option
# # tokens: A, B, C, D, E
# # lm acts as a wrapper for handling requests to OpenAi
# def lm(prompt,
#        max_tokens=256,
#        temperature=0,
#        logprobs=None,
#        stop_seq=None,
#        logit_bias={
#           317: 100.0,   #  A (with space at front)
#           347: 100.0,   #  B (with space at front)
#           327: 100.0,   #  C (with space at front)
#           360: 100.0,   #  D (with space at front)
#           412: 100.0,   #  E (with space at front)
#       },
#        timeout_seconds=20):
#   response = None
#   max_attempts = 5
#   for _ in range(max_attempts):
#       try:
#           with timeout(seconds=timeout_seconds):
#               response = openai.Completion.create(
#                   model='text-davinci-003',
#                   prompt=prompt,
#                   max_tokens=max_tokens,
#                   temperature=temperature,
#                   logprobs=logprobs,
#                   logit_bias=logit_bias,
#                   stop=list(stop_seq) if stop_seq is not None else None,
#               )
#           break
#       except:
#           print('Timeout, retrying...')
#           pass
#   if response is not None and "choices" in response and len(response["choices"]) > 0:
#     # Check if response is not None, has 'choices', and the choices list is not empty
#     text = response["choices"][0]["text"].strip()
#   else:
#      text = None  # Define a default value or error handling strategy
#   return response, text


# def process_mc_raw(mc_raw, add_mc='an option not listed here'):
#   mc_processed_all = []
#   mc_all = mc_raw.split('\n')
#   for mc in mc_all:
#       mc = mc.strip()

#       # skip nonsense
#       if len(mc) < 5 or mc[0] not in [
#           'a', 'b', 'c', 'd', 'A', 'B', 'C', 'D', '1', '2', '3', '4'
#       ]:
#           continue

#       mc = mc[2:]  # remove a), b), ...
#       mc = mc.strip().lower().split('.')[0]
#       mc_processed_all.append(mc)
#   if len(mc_processed_all) < 4:
#       raise 'Cannot extract four options from the raw output.'

#   # Check if any repeated option - use do nothing as substitute
#   mc_processed_all = list(set(mc_processed_all))
#   if len(mc_processed_all) < 4:
#       num_need = 4 - len(mc_processed_all)
#       for _ in range(num_need):
#           mc_processed_all.append('do nothing')
#   prefix_all = ['A) ', 'B) ', 'C) ', 'D) ']
#   if add_mc is not None:
#       mc_processed_all.append(add_mc)
#       prefix_all.append('E) ')
#   random.shuffle(mc_processed_all)

#   # combines the multiple choices into a single string
#   mc_prompt = ''
#   for mc_ind, (prefix, mc) in enumerate(zip(prefix_all, mc_processed_all)):
#       mc_prompt += prefix + mc
#       if mc_ind < len(mc_processed_all) - 1:
#           mc_prompt += '\n'
#   add_mc_prefix = prefix_all[mc_processed_all.index(add_mc)][0]
#   return mc_prompt, mc_processed_all, add_mc_prefix

# def temperature_scaling(logits, temperature):
#     logits = np.array(logits)
#     logits /= temperature

#     # apply softmax
#     logits -= logits.max()
#     logits = logits - np.log(np.sum(np.exp(logits)))
#     smx = np.exp(logits)
#     return smx

# mc_gen_prompt_template = """
# We: You are a robot, and you are asked to move objects to precise locations on the table. Our instructions can be ambiguous.

# We: On the table there are these objects: brown block, pink bowl, yellow block, green bowl, green block, blue bowl.
# We: Now, put the grass-colored bowl at the right side of the blue round object
# You: These are some options:
# A) put blue bowl at the right side of blue block
# B) put green bowl at the right side of blue bowl
# C) put green block at the right side of blue bowl
# D) put yellow bowl at the right side of blue bowl

# We: On the table there are these objects: yellow bowl, green bowl, green block, yellow block, blue block, blue bowl.
# We: Now, put the yellow square object near the green box
# You: These are some options:
# A) put yellow block in front of green block
# B) put yellow block behind green block
# C) put yellow block to the left of green block
# D) put yellow block to the right of green block

# We: On the table there are these objects: blue bowl, yellow block, green bowl, blue block, green block, yellow bowl.
# We: Now, put the yellow bowl along the horizontal axis of the grass-colored block
# You: These are some options:
# A) put yellow bowl at the front of the green block
# B) put yellow bowl at the left side of the green block
# C) put yellow bowl at the left side of the blue block
# D) put yellow bowl at the right side of the green block

# We: On the table there are these objects: green bowl, yellow block, blue bowl, yellow block, green block, blue bowl.
# We: Now, {instruction}
# You: These are some options:
# """

# mc_score_prompt_template = """
# We: You are a robot, and you are asked to move objects to precise locations on the table. Our instructions can be ambiguous.

# We: On the table there are these objects: green bowl, yellow block, blue bowl, yellow block, green block, blue bowl.
# We: Now, {instruction}
# You: These are some options:
# {mc}
# We: Which option is correct? Answer with a single letter.
# You:
# """
if __name__ == '__main__':
  instruction = input("Please input an instruction pertaining to the block in relation to bowl: ") #@param {type:"string"}
  skip_calibration = False #@param {type:"boolean"}
  verbose = False #@param {type:"boolean"}
  file_path = "/home/allen/catkin_ws/src/franka_irom/data/dataset.txt"
  if skip_calibration: qhat = 0.927 # based on epsilon=0.2
  else:
      with open(file_path,'r') as file:
          json_data = file.read()
          qhat = json.loads(json_data)[-1]['qhat']  #loads dataset list and reads off last dict element which stores calibration qhat

  # Get prompt for generating multiple choices
  mc_gen_prompt = mc_gen_prompt_template.replace('{instruction}', instruction).strip()
  if verbose:
    print('===== Prompt for generating multiple choices =====')
    print(mc_gen_prompt)

  # Generate multiple choices
  _, mc_gen_output = lm(mc_gen_prompt, logit_bias={})
  if verbose:
    print('\n===== Result =====')
    print(mc_gen_output)
  mc_gen_full, mc_gen_all, add_mc_prefix = process_mc_raw(mc_gen_output)

  # get new prompt
  mc_score_prompt = mc_score_prompt_template.replace('{instruction}', instruction).replace('{mc}', mc_gen_full).strip()
  if verbose:
    print('\n==== Prompt for LLM predicting next-token ====')
    print(mc_score_prompt)

  # call LLM API
  mc_score_response, _ = lm(mc_score_prompt, max_tokens=1, logprobs=5)
  top_logprobs_full = mc_score_response["choices"][0]["logprobs"]["top_logprobs"][0]
  top_tokens = [token.strip() for token in top_logprobs_full.keys()]
  top_logprobs = [value for value in top_logprobs_full.values()]



  # get the softmax value of true option
  mc_smx_all = temperature_scaling(top_logprobs, temperature=5)

  # true_label_smx = [mc_smx_all[token_ind]
  #                   for token_ind, token in enumerate(top_tokens)
  #                   if token in true_options]
  # true_label_smx = np.max(true_label_smx)

  # # get non-comformity score
  # non_conformity_score.append(1 - true_label_smx)

  # include all options with score >= 1-qhat
  prediction_set = [
            token for token_ind, token in enumerate(top_tokens)
            if mc_smx_all[token_ind] >= 1 - qhat
        ]

  # print
  print('\n===== Multiple choices generated =====')
  print(mc_gen_full)
  print('\n===== Prediction set =====')
  print("Ordered: ",prediction_set)
  if len(prediction_set) == 1:
    print('Singleton set. No help needed!')
  else:
    print('Set with multiple options. Help needed!')

  #@markdown If the prediction set has more than one option, human help is triggered. In that case, put the true option as a single capital letter here.
  user_chosen_option = input("Please pick a letter for the desired action in the prediction set:") #@param {type:"string"}
  action_option = mc_gen_all[sorted(top_tokens).index(user_chosen_option)]

  if len(prediction_set) > 1:
    print('Option chosen by user:', action_option)
  else:
    print('Option from prediction set without user intervention', )

  # extract pick and place locations
  if 'not listed here' in action_option or 'do nothing' in action_option or 'block ' not in action_option:
    print('Invalid option! Cannot execute.')
  else:
    # extract pick_obj from mc
    action_option_split = action_option.split()
    pick_obj_attr = action_option_split[action_option_split.index('block')-1]
    pick_obj = pick_obj_attr + ' block'

    # extract target_obj from mc
    target_obj_attr = action_option_split[action_option_split.index('bowl')-1]
    target_obj = target_obj_attr + ' bowl'

    # extract spatial relation from mc
    if 'left' in action_option:
      relation = 'left'
    elif 'right' in action_option:
      relation = 'right'
    elif 'front' in action_option:
      relation = 'front'
    elif 'back' in action_option or 'behind' in action_option:
      relation = 'back'
    else:
      relation = 'in'
    print('\nPick object:', pick_obj)
    print('Target object:', target_obj)
    print('Spatial relation to target object:', relation)


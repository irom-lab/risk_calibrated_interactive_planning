
import pickle
import openai
import signal
import random
import numpy as np
import random
import json


openai.api_key = "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"



# LLM API call
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

# OpenAI only supports up to five tokens (logprobs argument) for getting the likelihood.
# Thus we use the logit_bias argument to force LLM only consdering the five option
# tokens: A, B, C, D, E
# lm acts as a wrapper for handling requests to OpenAi
def lm(prompt,
       max_tokens=256,
       temperature=0,
       logprobs=None,
       stop_seq=None,
       logit_bias={
          317: 100.0,   #  A (with space at front)
          347: 100.0,   #  B (with space at front)
          327: 100.0,   #  C (with space at front)
          360: 100.0,   #  D (with space at front)
          412: 100.0,   #  E (with space at front)
      },
       timeout_seconds=20):
  response = None
  max_attempts = 5
  for _ in range(max_attempts):
      try:
          with timeout(seconds=timeout_seconds):
              response = openai.Completion.create(
                  model='text-davinci-003',
                  prompt=prompt,
                  max_tokens=max_tokens,
                  temperature=temperature,
                  logprobs=logprobs,
                  logit_bias=logit_bias,
                  stop=list(stop_seq) if stop_seq is not None else None,
              )
          break
      except:
          print('Timeout, retrying...')
          pass
  if response is not None and "choices" in response and len(response["choices"]) > 0:
    # Check if response is not None, has 'choices', and the choices list is not empty
    text = response["choices"][0]["text"].strip()
  else:
     text = None  # Define a default value or error handling strategy
  return response, text
#scenario distribution: Any changes in environment must be relected in colors, and prompt templates.
# Set up distribution
blocks = ['green block', 'brown block', 'yellow block']
bowls = ['green bowl', 'blue bowl', 'pink bowl']

instruction_ambiguities = {
    'put the block in the {color} bowl': 'block color',
    'put the {color} block in the bowl': 'bowl color',
    'put the {color} block in the {color} bowl': None,
    'put the {color} block close to the {color} bowl': 'direction',
    'put the {color} block to the {direction} of the {color} bowl': None
}
instruction_templates = list(instruction_ambiguities.keys()) # everything to list

colors = ['green', 'blue', 'yellow', 'brown', 'pink']
directions = ['front', 'back', 'left', 'right']

# parses and standardizes lm output
def process_mc_raw(mc_raw, add_mc='an option not listed here'):
  mc_processed_all = []
  mc_all = mc_raw.split('\n')
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

  # Check if any repeated option - use do nothing as substitute
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

  # combines the multiple choices into a single string
  mc_prompt = ''
  for mc_ind, (prefix, mc) in enumerate(zip(prefix_all, mc_processed_all)):
      mc_prompt += prefix + mc
      if mc_ind < len(mc_processed_all) - 1:
          mc_prompt += '\n'
  add_mc_prefix = prefix_all[mc_processed_all.index(add_mc)][0]
  return mc_prompt, mc_processed_all, add_mc_prefix

# converts logprobs to smx
def temperature_scaling(logits, temperature):
    logits = np.array(logits)
    logits /= temperature

    # apply softmax
    logits -= logits.max()
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return smx

def pickle_to_text_and_save(pickle_file_path, output_text_file_path):
    try:
        # Step 1: Load the pickle object
        with open(pickle_file_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        
        # Step 2: Convert the data to a text format (JSON in this case)
        text_data = json.dumps(data, indent=4)  # You can choose a different format if needed
        
        # Step 3: Save the text data into a file
        with open(output_text_file_path, 'w') as text_file:
            text_file.write(text_data)
        
        print(f'Pickle object successfully converted and saved as text in {output_text_file_path}')
    except Exception as e:
        print(f'An error occurred: {e}')

mc_gen_prompt_template = """
We: You are a robot, and you are asked to move objects to precise locations on the table. Our instructions can be ambiguous.

We: On the table there are these objects: green block, pink bowl, yellow block, green bowl, brown block, blue bowl.
We: Now, put the grass-colored block at the right side of the blue round object
You: These are some options:
A) put pink block at the right side of blue bowl
B) put green block at the right side of blue bowl
C) put green block at the right side of blue bowl
D) put yellow block at the right side of blue bowl

We: On the table there are these objects:  green block, pink bowl, yellow block, green bowl, brown block, blue bowl.
We: Now, put the yellow square object near the green box
You: These are some options:
A) put yellow block in front of green block
B) put yellow block behind green block
C) put yellow block to the left of green block
D) put yellow block to the right of green block

We: On the table there are these objects: green block, pink bowl, yellow block, green bowl, brown block, blue bowl.
We: Now, put the yellow bowl along the horizontal axis of the grass-colored block
You: These are some options:
A) put yellow bowl at the front of the green block
B) put yellow bowl at the left side of the green block
C) put yellow bowl at the left side of the blue block
D) put yellow bowl at the right side of the green block

We: On the table there are these objects: green block, pink bowl, yellow block, green bowl, brown block, blue bowl.
We: Now, {instruction}
You: These are some options:
"""

mc_score_prompt_template = """
We: You are a robot, and you are asked to move objects to precise locations on the table. Our instructions can be ambiguous.

We: On the table there are these objects: green block, pink bowl, yellow block, green bowl, brown block, blue bowl.
We: Now, {instruction}
You: These are some options:
{mc}
We: Which option is correct? Answer with a single letter.
You:
"""

if __name__ == '__main__':
    #Specify number of calibration data: with more calibration data, the guarantee from conformal prediction would be tighter and precise. We recommend at least 200 scenarios (400 used in the paper), but you can use a small number first to save the number of API calls.
    num_calibration = 400  
    
    #Sample from the distribution
    dataset = []
    for i in range(num_calibration):
      data = {}
      instruction_orig = random.choice(instruction_templates)
      instruction = instruction_orig
    
      # sample colors if needed
      num_color_in_instruction = instruction.count('{color}')
      if num_color_in_instruction > 0:
        color_instruction = random.choices(colors, k=num_color_in_instruction)
        for color in color_instruction:
          instruction = instruction.replace('{color}', color)
    
      # sample direction if needed
      if '{direction}' in instruction:
        direction = random.choice(directions)
        instruction = instruction.replace('{direction}', direction)
    
      # sample goal based on ambiguities
      ambiguity = instruction_ambiguities[instruction_orig]
      if ambiguity and 'color' in ambiguity:
        true_color = random.choice(colors)
      elif ambiguity and 'direction' in ambiguity:
        true_direction = random.choice(directions)
    
      # determine the goal in the format of [pick_obj, relation (in, left, right, front, back), target_obj]
      instruction_split = instruction.split()
      block_attr = instruction_split[instruction_split.index('block')-1]
      if 'the' == block_attr: # ambiguous
        pick_obj = true_color + ' block'
      else:
        pick_obj = block_attr + ' block'
      bowl_attr = instruction_split[instruction_split.index('bowl')-1]
      if 'the' == bowl_attr: # ambiguous
        target_obj = true_color + ' bowl'
      else:
        target_obj = bowl_attr + ' bowl'
      if 'next to' in instruction:
        relation = true_direction
      elif 'in' in instruction:
        relation = 'in'
      elif 'of' in instruction:
        relation = instruction_split[instruction_split.index('of')-1] # bit hacky
      else:
        relation = 'front' #front is used as default relation
    
      # fill in data
      data['environment'] = blocks + bowls # fixed set
      data['instruction'] = instruction
      data['goal'] = [pick_obj, relation, target_obj]
      dataset.append(data)
    
    # print a few
    print('Showing the first five sampled scenarios')
    for i in range(5):
      data = dataset[i]
      print(f'==== {i} ====')
      print('Instruction:', data['instruction'])
      print('Goal (pick_obj, relation, target_obj):', data['goal'])
    
    
    
    for i in range(len(dataset)):
      data = dataset[i]
      instruction = data['instruction']
      mc_gen_prompt = mc_gen_prompt_template.replace('{instruction}', instruction).strip()
    
      # call LLM API
      _, mc_gen_output = lm(mc_gen_prompt, logit_bias={})
      data['mc_gen_output'] = mc_gen_output # The generated mc_gen_output is associated with the current data item and stored back in the dataset.
      dataset[i] = data
    
    
    
    for i in range(len(dataset)):
      data = dataset[i]
      instruction = data['instruction']
      mc_gen_output = data['mc_gen_output']
    
      # process multiple choices
      mc_gen_full, mc_gen_all, add_mc_prefix = process_mc_raw(mc_gen_output)
      if i == 0:
        print('\n===== add_mc_prefix =====')
        print(add_mc_prefix)
    
      # get new prompt with instruction from current data and mc from mc_gen_full
      mc_score_prompt = mc_score_prompt_template.replace('{instruction}', instruction).replace('{mc}', mc_gen_full).strip()
      data['mc_score_prompt'] = mc_score_prompt
    
    
      # call LLM API
      mc_score_response, _ = lm(mc_score_prompt, max_tokens=1, logprobs=5)
      top_logprobs_full = mc_score_response["choices"][0]["logprobs"]["top_logprobs"][0]
      top_tokens = [token.strip() for token in top_logprobs_full.keys()]
      top_logprobs = [value for value in top_logprobs_full.values()]
    
      # save
      data['mc_gen_full'] = mc_gen_full
      data['mc_gen_all'] = mc_gen_all
      data['add_mc_prefix'] = add_mc_prefix
      data['mc_score_response'] = mc_score_response
      data['top_logprobs_full'] = top_logprobs_full
      data['top_tokens'] = top_tokens
      data['top_logprobs'] = top_logprobs
      dataset[i] = data
    
      # Simple heuristics to determine the true labels, i.e., the correct option from the multiple choices.
    for i, data in enumerate(dataset):
      mc_gen_all = data['mc_gen_all']
      goal = data['goal']
    
      # go through all mc
      token_all = ['A', 'B', 'C', 'D', 'E']
      true_labels = []
      for mc_ind, mc in enumerate(mc_gen_all):
        if 'not listed here' in mc or 'do nothing' in mc: continue
        if 'block ' not in mc: continue  # no object to be picked
        if 'bowl '  not in mc: continue
    
        # extract pick_obj from mc
        mc_split = mc.split()
        mc_pick_obj_attr = mc_split[mc_split.index('block')-1]
        mc_pick_obj = mc_pick_obj_attr + ' block'
    
        # extract target_obj from mc
        mc_target_obj_attr = mc_split[mc_split.index('bowl')-1]
        mc_target_obj = mc_target_obj_attr + ' bowl'
    
        # extract spatial relation from mc
        if 'left' in mc:
          relation = 'left'
        elif 'right' in mc:
          relation = 'right'
        elif 'front' in mc:
          relation = 'front'
        elif 'back' in mc or 'behind' in mc:
          relation = 'back'
        else:
          relation = 'in'
    
        # check with goal
        if mc_pick_obj == goal[0] and relation == goal[1] and mc_target_obj == goal[2]:
          true_labels.append(token_all[mc_ind])
    
      # if none correct
      if len(true_labels) == 0:
        true_labels = [data['add_mc_prefix']]
    
      # save
      dataset[i]['true_labels'] = true_labels
      
      # Calculating q-hat
      
    #Get the non-conformity scores from the calibration set, which is 1 minus the likelihood of the **true** option, $1-f(x)_{y_\text{true}}$.
    
    
    non_conformity_score = []
    for data in dataset:
      top_logprobs = data['top_logprobs']
      top_tokens = data['top_tokens']
      true_labels = data['true_labels']
    
      # normalize the five scores to sum of 1
      mc_smx_all = temperature_scaling(top_logprobs, temperature=5)
    
      # get the softmax value of true option
      true_label_smx = [mc_smx_all[token_ind]
                        for token_ind, token in enumerate(top_tokens)
                        if token in true_labels]
      true_label_smx = np.max(true_label_smx)
    
      # get non-comformity score
      non_conformity_score.append(1 - true_label_smx)
      
    # This is $1-\epsilon$ in the paper, the probability that the prediction set contains the true option at test time.
    target_success = 0.8  
    epsilon = 1-target_success
      
    q_level = np.ceil((num_calibration + 1) * (1 - epsilon)) / num_calibration  # quantile level
    qhat = np.quantile(non_conformity_score, q_level, method='higher')
    
    # Save qhat value as last dict at the end of dataset
    dataset.append({'qhat':qhat})
      
    # Pickle the list of dictionaries to a file 
    input = 'src/franka_irom/data/dataset.pkl'
    output = 'src/franka_irom/data/dataset.txt'
    
    with open(input, 'wb') as file:
        pickle.dump(dataset, file)
        
    pickle_to_text_and_save(input,output)

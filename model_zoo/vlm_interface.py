import openai
from openai import OpenAI
import signal
import tqdm.notebook as tqdm
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from os.path import expanduser

from utils.general_utils import str2bool

from environments.hallway_env import prompt

import base64
import requests

import argparse
import torch

from pathlib import Path
import time


from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from random import shuffle

from utils.vlm_utils import timeout, encode_image, response_pre_check

# from transformers import (
#     AutoProcessor,
#     LlavaConfig,
#     LlavaForConditionalGeneration,
#     is_torch_available,
#     is_vision_available,
# )

from PIL import Image
import unittest

# Set OpenAI API key.
openai_api_key = "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"  # Justin's key
openai.api_key = openai_api_key

prob_bins = list(np.arange(0, 101, 20))

NUM_INTENT_PILE = 5
NUM_PILE = 5
TOKENIZER_ABCDE_INDICES = [319, 350, 315, 360, 382]
TOKENIZER_ABC_INDICES = [319, 350, 315, 360]

# + f"For the <group_id> group of objects, what colors, functions, names, and sizes do you see? " \
PROMPT_VLM = ("The first three images are different bins sorted by a human. Describe the items in each of the bins."
              "For each bin, describe the color first, then the theme, then the shapes."
              # "The human sorts based on features such as color, shape, user, function, and theme."
              "The human first prefers to group red or orange items together based on color. All colors must be similar to match this theme." 
              "Look at the overall color of the objects, not at small details."
              "If the color does not match, the human prefers to group items by theme, such as fruits, vegetables, cookware, sauces, tableware, or children's toys. "
              "If the theme or color does not match, the human will group based on geometry, such as having a square shape."
              "The last image is the object we want to sort. Give a medium-length description for each bin and the object we want to sort."
              "Provide a ranking of the most-likely bins for the object."
              "Refer to the bins as Bin 1, Bin 2, and Bin 3."
              # "Next, give a short explanation for how the object could be sorted into each bin, assuming a method exists."
              "Use the words bin or objects to refer to what is in the picture. Don't use the word image." )
 
PROMPT_LLM = ("Here is a description of three bins and an object we want to sort.")

OPTIONS = ["Bin 1", "Bin 2", "Bin 3", "Unsure"]
LETTER_CHOICES = ["A", "B", "C", "D"]

PROMPT_DEBUG = "This is a set of four distinct images. What's in each image? List all objects you see."

def get_mcqa(include_none=True, shuffle_options=True, single_letter=True, python_dict_probs=False):

    mcqa = "Which bin should we place the object in?"
    num_options = len(OPTIONS)
    options_mcqa = list(OPTIONS)
    indices = list(range(len(options_mcqa)-1))
    if shuffle_options:
        shuffle(indices)
    indices = indices + [3]
    options_mcqa = [options_mcqa[i] for i in indices]
    for l, o in zip(LETTER_CHOICES[:num_options], options_mcqa):
        mcqa += f" {l}) {o} "

    # if include_none:
    #     mcqa += f" {LETTER_CHOICES[num_options]}) None of the above"

    if single_letter:
        mcqa += ". Give your answer as a single letter: A, B, C, or D."

    if python_dict_probs:
        mcqa += (f". Give your answer as a python dictionary with keys: {options_mcqa }. "
                 f"For each value, estimate your confidence in the answer with a floating-point number from 0 to 1."
                 "Ensure that the confidence scores sum to 1."
                 # f" try not to be too certain." 
                 # "You can indicate multiple bins with nonzero probabilities if multiple bins match different criteria."
                 "Just give the python dictionary, please")

    return mcqa, options_mcqa, indices


def get_prompt(group_descriptor="", debug=False, include_none=True, is_vlm=True):
    options_mcqa = indices = None
    if debug:
        ret_prompt = PROMPT_DEBUG
    elif is_vlm:
        ret_prompt = PROMPT_VLM
    else:
        mcqa, options_mcqa, indices = get_mcqa(include_none)
        ret_prompt = mcqa
    return ret_prompt, options_mcqa, indices

def next_alpha(s):
    return chr((ord(s.upper())+1 - 65) % 26 + 65)


def check_if_description_exists(image_files, temperature):
    description_filename = f'btest_new_description_single_token_temp_{temperature}.csv'

    description = None
    target_file_description = os.path.join(image_files, description_filename)
    if Path(target_file_description).is_file():
        description = pd.read_csv(target_file_description)["description"][0]

    return description

def check_if_plan_exists(image_files, temperature):
    plan_filename = f'btest_new_plan2_single_token_temp_{temperature}.csv'
    scores_filename = f'btest_new_scores2_single_token_temp_{temperature}.csv'

    plan = None
    scores = None

    try:
        target_file_plan = os.path.join(image_files, plan_filename)
        if Path(target_file_plan).is_file():
            plan = pd.read_csv(target_file_plan)["plan"][0]

        target_file_scores = os.path.join(image_files, scores_filename)
        if Path(target_file_scores).is_file():
            scores = pd.read_csv(target_file_scores)["scores"]
            scores = torch.Tensor(scores)
    except Exception as e:
        pass

    return plan, scores

def save_description(image_files, description, temperature):
    description_filename = f'btest_new_description_single_token_temp_{temperature}.csv'

    target_file_description = os.path.join(image_files, description_filename)

    description_df = pd.DataFrame({"description": [description]}, dtype=pd.StringDtype())

    description_df.to_csv(target_file_description)

    return

def save_plan(image_files, plan, scores, temperature):
    plan_filename = f'btest_new_plan2_single_token_temp_{temperature}.csv'
    scores_filename = f'btest_new_scores2_single_token_temp_{temperature}.csv'

    target_file_plan = os.path.join(image_files, plan_filename)
    target_file_scores = os.path.join(image_files, scores_filename)

    plan_df = pd.DataFrame({"plan": [plan]}, dtype=pd.StringDtype())
    scores_df = pd.DataFrame({"scores": scores.tolist()})

    plan_df.to_csv(target_file_plan)
    scores_df.to_csv(target_file_scores)

    return

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




def make_payload_vlm(prompt, image_dr, max_tokens, seed, temperature, is_dir=True):

    image_files = sorted(os.listdir(image_dr))
    image_files = [f for f in image_files if f.endswith("png")]
    image_files = image_files[1:] + image_files[0:1] # put the object last to match prompt
    full_image_paths = [os.path.join(image_dr, f) for f in image_files]
    base64_images = [encode_image(path) for path in full_image_paths]
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
        "temperature": temperature,
        "seed": seed,
        # "logprobs": True  # TODO: turn on when openAI releases this feature
    }
    return payload

def make_payload_llm(prompt, max_tokens, seed, temperature, is_dir=True):

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
        "seed": seed,
        "logprobs": True,
        "top_logprobs": 5
    }
    return payload


def generate_prediction_openai(prompt_vlm,
                               prompt_llm,
                               prompt_mcqa,
                               image_files,
                               indices,
                               max_tokens=1,
                               temperature_vlm=0,
                               temperature_llm=0,
                               seed=1234,
                               timeout_seconds=30,
                               max_attempts=100,
                               intent_set_size=12,
                               is_dir=True):
    text_description = check_if_description_exists(image_files, temperature_vlm)
    if text_description is None:
        text_description_raw, _ = vlm_or_llm(prompt_vlm, image_files, max_tokens=max_tokens,
                                             temperature=temperature_vlm, seed=seed, is_vlm=True)
        text_description = text_description_raw
        save_description(image_files, text_description, temperature_vlm)
    full_message, scores = check_if_plan_exists(image_files, temperature_llm)
    if full_message is None or scores is None:
        space = " "
        full_prompt = space.join([prompt_llm, text_description, prompt_mcqa])
        full_message, scores = vlm_or_llm(full_prompt, image_files, max_tokens=max_tokens,
                                          temperature=temperature_llm, seed=seed, is_vlm=False)
        scores = scores[indices]
        save_plan(image_files, full_message, scores, temperature_llm)
    return full_message, scores, text_description


# OpenAI only supports up to five tokens (logprobs argument) for getting the likelihood.
# Thus we use the logit_bias argument to force LLM only consdering the five option
# tokens: A, B, C, D, E
def vlm_or_llm(prompt,
        image_files,
       max_tokens=300,
       temperature=0.0,
       seed=1234,
       timeout_seconds=30,
       max_attempts=10,
       intent_set_size=12,
       is_dir=True,
       is_vlm=True):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    for _ in range(max_attempts):
        try:
            if is_vlm:
                payload = make_payload_vlm(prompt=prompt, max_tokens=max_tokens, temperature=temperature, seed=seed, image_dr=image_files)
            else:
                payload = make_payload_llm(prompt=prompt, max_tokens=1, temperature=temperature, seed=seed)
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            parsed_response = parse_response(response, logprobs_avail=(not is_vlm))
            break
        except Exception as e:
            print(e)
            e_type = "VLM" if is_vlm else "LLM"
            print(f"Error: {e_type}. Response below:")
            print(response.json())
            time.sleep(60)
    return parsed_response

def parse_response(response, logprobs_avail=False, python_dict_probs=False):
    resp_json = response.json()
    if "choices" in resp_json:
        response_json = resp_json["choices"][0]
        messages = response_json["message"]["content"]
    elif "error" in resp_json:
        response_json = resp_json["error"]
        messages = "A visual system error occurred. Please make your best guess."
    else:
        raise ValueError("Unexpected response.")
    logprobs = None
    if logprobs_avail:
        logprobs_tensor = -30*torch.ones(len(LETTER_CHOICES))
        top_logprobs = response_json["logprobs"]["content"][0]["top_logprobs"]
        token_prob_list = [(a['token'], a['logprob']) for a in top_logprobs]
        for t, l in token_prob_list:
            if t in LETTER_CHOICES:
                i = LETTER_CHOICES.index(t)
                logprobs_tensor[i] = l
        logprobs = logprobs_tensor
    elif python_dict_probs:
        d = messages.split('{')[-1]
        d = d.split('}')[0]
        choices = d.split(',')
        values = [float(c.split(':')[-1].strip()) for c in choices]
        values = torch.Tensor(values)/sum(values)
        logprobs = values.log()
    return messages, logprobs

def receptacles_parser(args):
    out = args.receptacles_dir.split(args.sep)[0]
    return out

def objects_parser(args):
    out = args.objects_dir.split(args.sep)[0]
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_dir):
    out = []
    for image_file in os.listdir(image_dir):
        full_file_path = os.path.join(image_dir, image_file)
        image = load_image(full_file_path)
        out.append(image)
    return out

def gather_files(image_dir):
    out = []
    for image_file in os.listdir(image_dir):
        full_file_path = os.path.join(image_dir, image_file)
        out.append(full_file_path)
    return out

def load_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    load_in_8bit = args.load_in_8bit
    load_in_4bit = args.load_in_4bit
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
    )
    return tokenizer, model, image_processor, model_name

def concatenate_images(images):
    images = [img.rotate(90) for img in images]
    height = max(img.height for img in images)
    total_width= sum(img.width for img in images) + 20 * (len(images) - 1)

    new_img = Image.new('RGB', (total_width, height), (0, 0, 0))

    current_height = 0
    for img in images:
        new_img.paste(img, (current_height, 0))
        current_height += img.width + 20  # adding a 20px black bar

    return new_img

def eval_model(tokenizer, model, image_processor, model_name, args, object_ind):


    use_start_end = False
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if use_start_end and model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if use_start_end and model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    receptacles_files = receptacles_parser(args)
    objects_files = objects_parser(args)
    images = load_images(receptacles_files)
    object_image = load_images(objects_files)[object_ind]
    images.append(object_image)
    images = concatenate_images(images) if len(images) > 1 else images[0]
    images = [images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        generation_output = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=[stopping_criteria],
        )

        input_token_len = input_ids.shape[1]
        output_ids = generation_output.sequences
        output_scores = generation_output.scores[0]

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs, output_scores

def query_model(tokenizer, model, image_processor, model_name, image_path, prompt, args, object_ind):

    args.query = prompt
    args.image_file = image_path
    preds, scores = eval_model(tokenizer, model, image_processor, model_name, args, object_ind)
    scores = scores[0, TOKENIZER_ABC_INDICES].softmax(-1)
    return preds, scores

def get_action_distribution_from_image(args, image_path, object_ind, temperature_llm, use_llava=False, llava_args=None, debug=False):

    if use_llava:
        tokenizer, model, image_processor, model_name = llava_args

    # receptacles_path = receptacles_parser(args)
    # objects_path = objects_parser(args)
    # receptacles_files = gather_files(receptacles_path)
    # objects_files = gather_files(objects_path)
    # files = receptacles_files
    # object_file = objects_files[object_ind]
    # files.append(object_file)

    # print(object_file)

    preds = []
    scores = []
    text = None
    prompt_mcqa, options, indices = get_prompt(debug=debug, is_vlm=False)
    loc_prompt = PROMPT_VLM + prompt_mcqa
    prompt_vlm = PROMPT_VLM
    prompt_llm = PROMPT_LLM
    if use_llava:
        pred, score = query_model(tokenizer, model, image_processor, model_name, image_path, loc_prompt, args, object_ind)
    else:
        pred, score, text = generate_prediction_openai(prompt_vlm,
                                                 prompt_llm,
                                                 prompt_mcqa,
                                                 image_path,
                                                 indices,
                                                 temperature_vlm=args.temperature,
                                                 temperature_llm=temperature_llm,
                                                 max_tokens=args.max_new_tokens)
        # score = score[indices]
    # print(pred)
    # print(score)
    # print(options)
    # if debug:
    #     print(pred)
    return pred, score.cpu(), text

# def processed_probs(file_name, t=0):
#
#     ground_truth_file = os.path.join(file_name, "ground_truth.csv")
#     df = pd.read_csv(ground_truth_file)
#     if "action_set_probs" in df:
#         return torch.Tensor(df["action_set_probs"][t])
#     return None
#
# def save_processed_probs(file_name, probs):
#
#     ground_truth_file = os.path.join(file_name, "ground_truth.csv")
#     df = pd.read_csv(ground_truth_file)
#     df["action_set_probs"] = probs.cpu().tolist()
#     #df.to_csv(ground_truth_file)
#     return None


if __name__ == "__main__":
    # home = expanduser("~")
    # image_path = os.path.join(home, 'PredictiveRL/franka_img_test/blocks.png')
    # qa, alpha_ids = gen_prompt_qa()
    # qa_str = '\n'.join(qa)
    # prompt_full = prompt_pre + "\n" + qa_str + " \n" + prompt_suffix
    # print(prompt_full)
    #
    # response = vlm(prompt_full, image_path, max_tokens=300, seed=1234, is_dir=False, intent_set_size=12)
    # probs = parse_response(response)
    # print(probs)
    # test_small_model_integration_test_llama_batched()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--receptacles-dir", type=str, default="/home/jlidard/PredictiveRL/language_img/receptacles")
    parser.add_argument("--objects-dir", type=str, default="/home/jlidard/PredictiveRL/language_img/objects")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    parser.add_argument("--load-in-8bit", type=str2bool, default=True)
    parser.add_argument("--load-in-4bit", type=str2bool, default=False)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--use-llava", type=str2bool, default=False)

    args = parser.parse_args()
    debug = args.debug
    use_llava = args.use_llava

    llava_args = None

    if use_llava:
        tokenizer, model, image_processor, model_name = load_model(args)
        llava_args = tokenizer, model, image_processor, model_name


    img_path = "/home/jlidard/PredictiveRL/franka_calibration/calibration/calibration_545/time_0"
    print(get_prompt())
    correct_count = 0
    for object_ind in range(10):
        object_ind = 2
        preds, scores, text = get_action_distribution_from_image(img_path,
                                                                 object_ind=object_ind,
                                                                 use_llava=use_llava,
                                                                 debug=debug,
                                                                 llava_args=llava_args)
        # if 'A' in preds:
        #     correct_count += 1
    print(correct_count)
    print(preds)
    print(scores)

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

from utils.general_utils import str2bool

from environments.hallway_env import prompt

import base64
import requests

import argparse
import torch


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

prompt = f"Here is an image of a collection of common 3D shapes sorted by a human. \n " \
    + "Each group can only be sorted by a single property (e.g. yellow and green objects don't work for property: color)." \
    + f"For the <group_id> group of objects, explain the sorting method: \n" \
    + "A) Color \nB) Function \nC) Name  \nD) Size \nE) None of the above.\n " \
    + ". \n"

prompt_debug = "What's in this image?"

def get_prompt(group_descriptor="", debug=False):
    if debug:
        ret_prompt = prompt_debug
    else:
        ret_prompt = prompt.replace("<group_id>", group_descriptor)
    return ret_prompt

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
    print(response_str)
    probs = re.findall(r"[-+]?(?:\d*\.*\d+)%", response_str)
    probs = [float(x.split(':')[0]) for x in probs]
    return np.array(probs)

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
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

def eval_model(tokenizer, model, image_processor, model_name, args):


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

    image_files = image_parser(args)
    images = load_images(image_files)
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

def query_model(tokenizer, model, image_processor, model_name, image_path, prompt, args):

    args.query = prompt
    args.image_file = image_path
    preds, scores = eval_model(tokenizer, model, image_processor, model_name, args)
    return preds, scores

def get_intent_distribution_from_image(image_path, debug=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load-in-8bit", type=str2bool, default=True)
    parser.add_argument("--load-in-4bit", type=str2bool, default=False)

    args = parser.parse_args()
    tokenizer, model, image_processor, model_name = load_model(args)

    obj_locations = ["upper_left", "upper_right", "center", "lower_left", "lower_right"]
    intent_probs = torch.zeros((NUM_INTENT_PILE, NUM_INTENT_PILE))
    preds = []
    scores = []
    for i, loc in enumerate(obj_locations):
        loc_prompt = get_prompt(group_descriptor=loc, debug=debug)
        pred, score = query_model(tokenizer, model, image_processor, model_name, image_path, loc_prompt, args)
        if debug:
            print(pred)
        score = score[0].cpu() # ignore batch dim
        top_indices = torch.Tensor(TOKENIZER_ABCDE_INDICES).long() # ABCDE token indices
        top_scores = score[top_indices]
        intent_probs[i, :] += top_scores
        preds.append(pred)
        scores.append(top_scores)

    return intent_probs, preds, scores



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


    img_path = "/home/jlidard/PredictiveRL/IMG_5008.jpg"
    debug=False

    intent_probs, preds, scores = get_intent_distribution_from_image(img_path, debug=debug)
    print(intent_probs)
    print(preds)
    print(scores)

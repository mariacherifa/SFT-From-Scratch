#####################################################################
#                           Prepare Data 
#####################################################################
import json 
import torch 
from tokenizer import encode, decode 

def load_sft_data(file_path):
    with open(file_path, "r", encoding = "utf-8") as f: 
        raw_text = json.load(f)

    return raw_text 

def format_example(example):

    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    if input_text:
        prompt = (
            f"I:{instruction}\n"
            f"X:{input_text}\n"
            f"A:"
        )
    else:
        prompt = (
            f"I:{instruction}\n"
            f"A:"
        )


    return prompt, output

def build_sft_text(examples):
    full_text = ""

    for example in examples:
        prompt, response = format_example(example)
        full_text += prompt + response

    return full_text

def check_unknown_characters(examples, stoi):
    unknown = set()

    for example in examples:
        text = format_example(
            instruction=example["instruction"],
            response=example["output"],
            input_text=example.get("input", "")
        )

        for character in text:
            if character not in stoi:
                unknown.add(character)

    return unknown

def filter_compatible_examples(examples, stoi):

    compatible_examples = []
    removed_examples = []

    for example in examples:

        prompt, response = format_example(example)
        text = prompt + response

        unknown = {
            character
            for character in text
            if character not in stoi
        }

        if len(unknown) == 0:
            compatible_examples.append(example)

        else:
            removed_examples.append({
                "example": example,
                "unknown_characters": sorted(unknown)
            })

    return compatible_examples, removed_examples



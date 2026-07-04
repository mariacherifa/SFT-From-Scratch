# ==================================================
#                 Prompt Formatting 
# ==================================================

import json 

# Loading json file 

def load_json(p): 
    with open(p, "r", encoding= "utf-8") as f :
        examples = json.load(f)
    return examples

# Formatting examples 

def format_example(instruction, response = None):
    if response is None : 
        text = f"Instruction:\n{instruction}\n\nResponse:\n"
    else:
        text = f"Instruction:\n{instruction}\n\nResponse:\n{response}"
    return text 

# SFT text 

def build_sft_text(examples):
    full_text = ""
    for example in examples:
        full_text += format_example(example["instruction"], example["response"])

    return full_text


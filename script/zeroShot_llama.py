import argparse
import random

import numpy as np
import pandas as pd
import sys
import re

from peft import PeftModel
from rank_bm25 import BM25Okapi
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import time

sys.path.append("..")

from utils.universal_utils import most_similar_item,get_dataset_classes
import transformers
import torch
# Load model directly
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import os
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def generate_answer(model, descriptions, classes):
    prompt = [prompter.generate_prompt(descriptions, "", class_item) for class_item in classes]
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    prompt_no_class = prompter.generate_prompt(descriptions,"")
    input_no_class = tokenizer.encode(prompt_no_class,return_tensors="pt")[0].to(device)

    no_class_len = len(input_no_class)
    labels = input_ids.cpu().numpy().tolist()
    for class_index in range(len(classes)):
        begin_index = 0
        for i in range(20):
            if torch.equal(input_ids[class_index][i:i+no_class_len],input_no_class):
                begin_index = i
                break

        labels[class_index][:begin_index+no_class_len] = [-100] *  (begin_index+no_class_len)
    labels = torch.tensor(labels).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='sum')
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_list = [loss_fct(shift_logits[i], shift_labels[i]) for i in range(len(classes))]
    min_class_index = loss_list.index(min(loss_list))
    return classes[min_class_index]

def generate_full_answer(model, prompt, pbar, new_tokens=10):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        # do_sample=True,
    )

    with torch.no_grad():
        # need to rewrite:
        model_inputs = {"input_ids":input_ids,"attention_mask":attention_mask,"labels":input_ids}
        outputs = model(**model_inputs)
        next_token_logits = outputs.logits[:, -1, :]

        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=new_tokens,
        )
    s = generation_output.sequences[0]
    score = generation_output.sequences_scores.item()
    # pbar.write(f"score:{score}")
    output = tokenizer.decode(s,skip_special_tokens=True)
    return output, score


parser =  argparse.ArgumentParser()
parser.add_argument('--base_model')
parser.add_argument('-t','--target_data_path',nargs='+')
parser.add_argument('-p','--prompt_template_name')
parser.add_argument('--dataset_name',default='Office31')
parser.add_argument('--class_eval',action="store_true")
parser.add_argument('--save_path')
parser.add_argument('--lora_weights')
args = parser.parse_args()



# load model
model_path = args.base_model
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)

classes = get_dataset_classes(args.dataset_name)
classes_token = [tokenizer.encode(class_item) for class_item in classes]

model = LlamaForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
)
if args.lora_weights:
    print("Load peft models...")
    model = PeftModel.from_pretrained(
        model,
        args.lora_weights,
        torch_dtype=torch.float16,
    )

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.half()  # seems to fix bugs for some users.

model.eval()

# get the source_descriptions
if args.dataset_name == 'subDomainNet':
    target_datas = []
    for path in args.target_data_path:
        target_data = pd.read_csv(path)
        target_data = target_data[target_data['categories'].isin(classes)]
        target_datas.append(target_data)
else:
    target_datas = [pd.read_csv(path) for path in args.target_data_path]
target_data = pd.concat(target_datas)

target_data = target_data.sample(frac=1, random_state=1024)

# source_data_len = len(source_data)
# print("source data len:", source_data_len)
target_data_len = len(target_data)
print("target data len:", target_data_len)

prompter = Prompter(args.prompt_template_name)

def get_prompt_split(description):
    key_words = ["Tags:","Attributes:","Captions:"]
    components = [description]
    for key_word in key_words:
        temp = []
        for component in components:
            temp.extend(component.split(key_word))
        components = temp
    return components[1:]

# beging test:
correct = 0

threshold = -0.0002
# right_sample_score = []
# wrong_sample_score = []
samples_output = []
pbar = tqdm(range(target_data_len))
per_class_correct = None

if args.class_eval:
    per_class_correct = {item:{"correct":0, "count": 0} for item in classes}

begin_time = time.time()
for i in pbar:
    item = target_data.iloc[i]
    target = most_similar_item(item.categories, classes)
    answer = generate_answer(model, item.descriptions, classes)
    # output = most_similar_item(answer, classes)
    output = answer

    sample = {'categories':target, 'pseudo_label': output, 'descriptions': item.descriptions}
    samples_output.append(sample)

    if output!=target:
        pbar.write(f"!!!!! Different")
    pbar.write(f"{i}:output1:{output}, target:{target}")

    if args.class_eval:
        per_class_correct[target]['count'] += 1

    if target == output:
        correct += 1
        if args.class_eval:
            per_class_correct[target]['correct'] += 1

    pbar.set_postfix({"acc":f"{correct / (i + 1) * 100 : .2f}"})


# end_time = time.time()
# all_time = end_time - begin_time
# print(all_time/100)
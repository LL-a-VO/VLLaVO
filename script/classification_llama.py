import argparse
import random
import time

import numpy as np
import pandas as pd
import sys
import re

from peft import PeftModel
from rank_bm25 import BM25Okapi
from tqdm import tqdm

sys.path.append("..")

from utils.universal_utils import most_similar_item,get_dataset_classes
import transformers
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import os
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def generate_full_answer(model, prompt, pbar, new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        # do_sample=True,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=new_tokens,
        )
    s = generation_output.sequences[0]
    score = generation_output.sequences_scores.item()
    output = tokenizer.decode(s,skip_special_tokens=True)
    return output, score


parser =  argparse.ArgumentParser()
parser.add_argument('--base_model')
parser.add_argument('--lora_weights')
parser.add_argument('-s','--source_data_path')
parser.add_argument('-t','--target_data_path',nargs='+')
parser.add_argument('--pseudo_label_path')
parser.add_argument('-p','--prompt_template_name')
parser.add_argument('--dataset_name',default='Office31')
parser.add_argument('--class_eval',action="store_true")
parser.add_argument('--save_path')
args = parser.parse_args()

classes = get_dataset_classes(args.dataset_name)

# load model
model_path = args.base_model
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
)
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
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

# get the source_descriptions
# source_data = pd.read_csv(args.source_data_path)
if args.dataset_name == 'subDomainNet':
    target_datas = []
    for path in args.target_data_path:
        target_data = pd.read_csv(path)
        target_data = target_data[target_data['categories'].isin(classes)]
        target_datas.append(target_data)
else:
    target_datas = [pd.read_csv(path) for path in args.target_data_path]
target_data = pd.concat(target_datas)

# only target data need to be shuffled
if args.pseudo_label_path:
    pseudo_labels = np.load(args.pseudo_label_path)
    target_data['pseudo_labels'] = pseudo_labels

target_data = target_data.sample(frac=1, random_state=1024)

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

correct = 0

threshold = -0.0002
samples_output = []
pbar = tqdm(range(target_data_len))
per_class_correct = None

if args.class_eval:
    per_class_correct = {item:{"correct":0,"count":0} for item in classes}

begin_time = time.time()
for i in pbar:
    if i> 500:
        break
    item = target_data.iloc[i]
    target = most_similar_item(item.categories, classes)

    prompt = prompter.generate_prompt(item.descriptions,"")
    full_answer,score = generate_full_answer(model,prompt,pbar)
    answer = prompter.get_response(full_answer)
    output = answer

    sample = {'categories':target,'pseudo_label':output,'score':score,'descriptions':item.descriptions}
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

end_time = time.time()
all_time = end_time - begin_time
print(all_time/500)

if args.save_path:
    pd.DataFrame(samples_output).to_csv(args.save_path)
if args.class_eval:
    class_avg = {item: per_class_correct[item]['correct'] / per_class_correct[item]['count'] * 100
                 for item in per_class_correct}
    class_avg = pd.DataFrame(class_avg)
    print(class_avg)
import os
import sys

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

sys.path.append('..')
from typing import List

import numpy as np
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from utils.universal_utils import most_similar_item, get_dataset_classes

import logging

logging.basicConfig(level=logging.INFO)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainerCallback, TrainingArguments, TrainerState, \
    TrainerControl, GenerationConfig


from utils.prompter import Prompter

# 自定义dataset获取方法，返回DatasetDict
def get_dataset(data_path):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    elif data_path.endswith(".csv"):
        data = load_dataset("csv", data_files=data_path)
    else:
        data = load_dataset(data_path)
    return data

def get_dataset_pseudo(data_path):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    elif data_path.endswith(".csv"):
        data = load_dataset("csv", data_files=data_path)
    else:
        data = load_dataset(data_path)
    data['train'] = data['train'].remove_columns("categories")

    return data

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    source_data_path: str = "../datasets/dslr_description.csv",
    target_data_path: str = "",
    pseudo_label_path: str = "",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 0,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    used_components: str = "tac",
    dataset_name: str = 'OfficeHome'
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"source_data_path: {source_data_path}\n"
            f"target_data_path: {target_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n" #一张卡的batch_size.
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    classes = get_dataset_classes(dataset_name)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def extract_modules(text):
        extracted_data = {}
        cur_module = ""
        for text in text.split("\n"):
            if text:
                if text[-1] == ":":
                    cur_module = text[:-1]
                    extracted_data[cur_module] = []
                else:
                    extracted_data[cur_module].append(text)
        return extracted_data

    def refine_descriptions(descriptions):
        refined_descriptions = ""
        deconstruct_prompt = extract_modules(descriptions)
        if 't' in used_components:
            refined_descriptions += "Tags:\n"
            refined_descriptions += "\n".join(deconstruct_prompt["Tags"]) + "\n"
        if 'a' in used_components:
            refined_descriptions += "Attributes:\n"
            refined_descriptions += "\n".join(deconstruct_prompt["Attributes"]) + "\n"
        if 'c' in used_components:
            refined_descriptions += "Captions:\n"
            refined_descriptions += "\n".join(deconstruct_prompt["Captions"])
        return refined_descriptions

    def generate_and_tokenize_prompt(data_point):
        descriptions = data_point['descriptions']
        descriptions = refine_descriptions(descriptions)
        full_prompt = prompter.generate_prompt(
            data_point["descriptions"],
            "",
            most_similar_item(data_point["categories"],classes),
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["descriptions"], ""
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt



    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    source_data = get_dataset(source_data_path)
    target_data = get_dataset_pseudo(target_data_path)
    target_data = target_data['train'].rename_column('pseudo_label','categories')
    # threshold = -0.0002
    # target_data = target_data.sort('score',reverse=True)
    # target_data = target_data.select(range(int(len(target_data)*0.5)))
    # target_data = target_data.filter(lambda datapoint: datapoint['score'] > threshold)
    source_data = concatenate_datasets([source_data['train'],target_data])
    # source_data = target_data

    # 准备training data
    data_transform = generate_and_tokenize_prompt
    if val_set_size > 0:
        train_val = source_data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(data_transform)
        )
        val_data = (
            train_val["test"].shuffle().map(data_transform)
        )
    else:
        train_data = source_data.shuffle().map(data_transform)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    print("gradient_accumulation_steps:", gradient_accumulation_steps)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        # compute_metrics=compute_metrics,
        args=transformers.TrainingArguments(
            disable_tqdm=True,
            # log_level='debug',
            logging_first_step=True,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.1,
            # warmup_steps=100,
            dataloader_drop_last=False,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True, #v100 didn't support bf16
            logging_steps=1, # steps means the optimized step.
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=1 if val_set_size > 0 else None,
            save_steps=20,
            output_dir=output_dir,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )



if __name__ == "__main__":
    # with torch.autocast("cuda"):
    fire.Fire(train)

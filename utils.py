import os
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaForCausalLM
from utils import DataPreparer, all_exists
from peft import PeftModel, prepare_model_for_int8_training
from typing import Dict, Optional, Callable
from functools import partial

def get_translation_template(dataset, **formatting_kwargs):
    base_template = open(f'./templates/translation/{dataset}.txt', 'r').read()
    template = base_template.format(**formatting_kwargs)
    return template

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": '<unk>',
        "pad_token": '<unk>',
    })
    tokenizer.padding_side = "left"  # Allow batched inference

    model = LlamaForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b',
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map='auto',
    )

    model = prepare_model_for_int8_training(model)

    model = PeftModel.from_pretrained(
        model,
        'yuan-yang/LogicLLaMA-7b-direct-translate-delta-v0',
        torch_dtype=torch.float16
    )

    model.to(device)

    return model, tokenizer


def query_model(input_prompt, dataset, model, tokenizer):
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1
    )
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
    input_ids = input_ids.to(model.device)
    output = model.generate(input_ids, **generation_config)
    return tokenizer.decode(output[0], skip_special_tokens=True)

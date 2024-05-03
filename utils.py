import os
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, prepare_model_for_kbit_training
from dotenv import load_dotenv
import re
from huggingface_hub import login
load_dotenv()

login(token=os.getenv('HF_TOKEN'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
translation_base_model = 'meta-llama/Llama-2-7b-chat-hf'
translation_peft_model = 'yuan-yang/LogicLLaMA-7b-direct-translate-delta-v0.1'
inference_base_model = 'microsoft/Phi-3-mini-4k-instruct'

def parse_fol(llama_output):
    fol_translated = llama_output.split('### FOL:')[-1].strip()
    return fol_translated

def parse_inference(phi_output):
    return phi_output.split('### Answer')[-1].strip()
    # complete_answer = phi_output.split('### Answer')[-1].strip()
    # answer_regex = r'(\w)\)'
    # try:
    #     answer = re.search(answer_regex, complete_answer).group(0)
    # except:
    #     answer = ''
    #     print(f'Error parsing answer: {complete_answer}')
    # return complete_answer, answer 

def get_template(task, dataset, **formatting_kwargs):
    base_template = open(f'./templates/{task}/{dataset}.txt', 'r').read()
    template = base_template.format(**formatting_kwargs)
    return template

def load_translation_model():
    tokenizer = LlamaTokenizer.from_pretrained(translation_base_model)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": '<unk>',
        "pad_token": '<unk>',
    })
    tokenizer.padding_side = "left"  # Allow batched inference

    bitsandbytes_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_compute_dtype=torch.float16,

    )      
    model = LlamaForCausalLM.from_pretrained(
        translation_base_model,
        quantization_config=bitsandbytes_config,
        torch_dtype=torch.float16,
        device_map='auto',
    )

    model = prepare_model_for_kbit_training(model)

    model = PeftModel.from_pretrained(model, model_id=translation_peft_model,torch_dtype=torch.float16)

    model.to(device)

    return model, tokenizer

def load_inference_model():
    tokenizer = AutoTokenizer.from_pretrained(inference_base_model)
    model = AutoModelForCausalLM.from_pretrained(inference_base_model, trust_remote_code=True)
    model.to(device)
    return model, tokenizer


def query_model(input_prompt, model, tokenizer, task='translation'):
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=1,
        do_sample=True,
        num_beams=1,
    )
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
    input_ids = input_ids.to(model.device)
    model.eval()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=128 if task == 'translation' else 32,
            )
    model.train()
    output_ids = generation_output.sequences[0]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output_text

def query_translation_model(input_prompt, model, tokenizer):
    output_text = query_model(input_prompt, model, tokenizer)
    return parse_fol(output_text)

def query_inference_model(input_prompt, model, tokenizer):
    output_text = query_model(input_prompt, model, tokenizer)
    return parse_inference(output_text)
import os
import torch
import re
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from huggingface_hub import login

load_dotenv()
login(token=os.getenv('HF_TOKEN'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

translation_base_model = 'meta-llama/Llama-2-7b-chat-hf'
translation_peft_model = 'yuan-yang/LogicLLaMA-7b-direct-translate-delta-v0.1'

inference_base_model = 'microsoft/Phi-3-mini-128k-instruct'

def parse_fol(llama_output):
    fol_translated = llama_output.split('### FOL:')[-1].strip()
    return fol_translated

def parse_inference(phi_output):
    return phi_output.split('### Answer')[-1].strip()

def get_template(task, template_filename, formatting_kwargs):
    base_template = open(f'./templates/{task}/{template_filename}.txt', 'r').read()
    template = base_template.format(**formatting_kwargs)
    return template

def load_vllm_translation_model():
    model = LLM('LorMolf/LogicLlama2-chat-direct', dtype='auto')
    return model

# def load_vllm_inference_model():
#     model = LLM(inference_base_model, dtype='auto', trust_remote_code=True)
#     return model
    
def query_vllm_model(model, input_prompts, max_tokens=128):
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens, n=1)
    output_text = model.generate(input_prompts, sampling_params=sampling_params)
    texts = [output.outputs[0].text.strip() for output in output_text]
    return texts


def load_inference_model():
    tokenizer = AutoTokenizer.from_pretrained(inference_base_model)
    model = AutoModelForCausalLM.from_pretrained(inference_base_model, trust_remote_code=True, torch_dtype=torch.float16)
    model.eval()
    model.to(device)
    return model, tokenizer


def query_model(tokenizer, model, input_prompt, max_tokens=64):
    input_ids = tokenizer(input_prompt, return_tensors='pt', padding=True).input_ids
    input_ids = input_ids.to(device)

    gen_config = GenerationConfig(
        max_new_tokens = max_tokens
    )

    output = model.generate(input_ids, generation_config=gen_config)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text
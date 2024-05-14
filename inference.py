import click
from utils import get_template, load_inference_model, query_model
import json
from tqdm import tqdm

def commonsense_qa(input_filename, output_filename, template_filename):
    with open(f'outputs/cqa/{input_filename}', 'r', encoding='utf-8') as f:
        cqa = json.load(f)
    model, tokenizer = load_inference_model()
    for i, item in enumerate(tqdm(cqa)):
        nl_question = item['nl_question']
        fol_question = item['fol_question']

        possible_answers = '\n'.join(item['possible_answers'])
        
        input_prompt = get_template(task='inference', dataset='cqa', template_filename=template_filename, formatting_kwargs={
            'nl_question': nl_question,
            'fol_question': fol_question,
            'possible_answers': possible_answers
        })
        
        model_output = query_model(tokenizer=tokenizer, model=model, input_prompt=input_prompt, max_tokens=8)
        model_answer = model_output.split('### Answer')[-1].strip()
        item['model_output'] = model_output
        item['model_answer'] = model_answer
        
    with open(f'outputs/cqa/{output_filename}', 'w',  encoding='utf-8') as f:
        json.dump(cqa, f, indent=4, ensure_ascii=False)

def hotpotqa(input_filename, output_filename, template_filename):
    with open(f'outputs/hqa/{input_filename}', 'r', encoding='utf-8') as f:
        hqa = json.load(f)
    model, tokenizer = load_inference_model()
    for i, item in enumerate(tqdm(hqa)):
        nl_question = item['nl_question']
        fol_question = item['fol_question']

        nl_contexts = '\n'.join(item['nl_contexts'])
        fol_contexts = '\n'.join(item['fol_contexts'])
        
        input_prompt = get_template(task='inference', dataset='hqa', template_filename=template_filename, formatting_kwargs={
            'fol_contexts': fol_contexts,
            'fol_question': fol_question,
        })
        
        model_output = query_model(tokenizer=tokenizer, model=model, input_prompt=input_prompt, max_tokens=8)
        model_answer = model_output.split('### Answer')[-1].strip()
        item['model_output'] = model_output
        item['model_answer'] = model_answer

    with open(f'outputs/hqa/{output_filename}', 'w', encoding='utf-8') as f:
        json.dump(hqa, f, indent=4, ensure_ascii=False)

def openbookqa(input_filename, output_filename, template_filename):
    with open(f'outputs/obqa/{input_filename}', 'r', encoding='utf-8') as f:
        obqa = json.load(f)
        
    model, tokenizer = load_inference_model()
    for i, item in enumerate(tqdm(obqa)):
        nl_question = item['nl_question']
        fol_question = item['fol_question']

        nl_context = item['nl_context']
        fol_context = item['fol_context']

        possible_answers = '\n'.join(item['possible_answers'])

        input_prompt = get_template(task='inference', dataset='obqa', template_filename=template_filename, formatting_kwargs={
            'fol_context': fol_context,
            'fol_question': fol_question,
            'possible_answers': possible_answers
        })

        model_output = query_model(tokenizer=tokenizer, model=model, input_prompt=input_prompt, max_tokens=8)
        model_answer = model_output.split('### Answer')[-1].strip()
        item['model_output'] = model_output
        item['model_answer'] = model_answer

    with open(f'outputs/obqa/{output_filename}', 'w', encoding='utf-8') as f:
        json.dump(obqa, f, indent=4, ensure_ascii=False)

def cqa20(input_filename, output_filename, template_filename):
    with open(f'outputs/cqa20/{input_filename}', 'r', encoding='utf-8') as f:
        cqa20 = json.load(f)
    model, tokenizer = load_inference_model()
    for i, item in enumerate(tqdm(cqa20)):
        nl_question = item['nl_question']
        fol_question = item['fol_question']

        correct_answer = item['correct_answer']
        possible_answers = 'A) yes\nB) no'

        input_prompt = get_template(task='inference', dataset='cqa20', template_filename=template_filename, formatting_kwargs={
            'fol_question': nl_question,
            'possible_answers': possible_answers,
        })

        model_output = query_model(tokenizer=tokenizer, model=model, input_prompt=input_prompt, max_tokens=8)
        model_answer = model_output.split('### Answer')[-1].strip()
        item['model_output'] = model_output
        item['model_answer'] = model_answer

    with open(f'outputs/cqa20/{output_filename}', 'w', encoding='utf-8') as f:
        json.dump(cqa20, f, indent=4, ensure_ascii=False)

def snli(input_filename, output_filename, template_filename):
    with open(f'outputs/snli/{input_filename}', 'r', encoding='utf-8') as f:
        snli = json.load(f)
    model, tokenizer = load_inference_model()
    for i, item in enumerate(tqdm(snli)):
        nl_premise = item['nl_premise']
        nl_hypothesis = item['nl_hypothesis']

        fol_premise = item['fol_premise']
        fol_hypothesis = item['fol_hypothesis']

        possible_answers = item['possible_answers']
        correct_answer = item['correct_answer']

        input_prompt = get_template(task='inference', dataset='snli', template_filename=template_filename, formatting_kwargs={
            'fol_premise': fol_premise,
            'fol_hypothesis': fol_hypothesis,
            'possible_answers': possible_answers
        })

        model_output = query_model(tokenizer=tokenizer, model=model, input_prompt=input_prompt, max_tokens=8)
        model_answer = model_output.split('### Answer')[-1].strip()
        item['model_output'] = model_output
        item['model_answer'] = model_answer

    with open(f'outputs/snli/{output_filename}', 'w', encoding='utf-8') as f:
        json.dump(snli, f, indent=4, ensure_ascii=False)

@click.command()
@click.option('--dataset', '-d', help='Dataset to be used', type=click.Choice(['cqa', 'hqa', 'obqa', 'cqa20', 'snli']))
@click.option('--input', '-i', type=str, help='Dataset to perform inference on')
@click.option('--output', '-o', type=str, help='Output filename')
@click.option('--template', '-t', type=str, help='Template filename for inference prompt')
def inference(dataset, input, output, template):
    callables = {
        'cqa': commonsense_qa,
        'hqa': hotpotqa,
        'obqa': openbookqa,
        'cqa20': cqa20,
        'snli': snli
    }

    if dataset in callables:
        callables[dataset](input, output, template)    
    else:
        raise ValueError('Dataset not found')


if __name__ == '__main__':
    inference()
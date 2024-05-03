import click
from utils import get_template, load_inference_model, query_inference_model
import json
from tqdm import tqdm

def commonsense_qa(input_filename, output_filename):
    with open(f'outputs/{input_filename}.json', 'r') as f:
        cqa = json.load(f)
    model, tokenizer = load_inference_model()
    for i, item in enumerate(tqdm(cqa)):
        # nl_question = item['nl_question']
        fol_question = item['fol_question']

        possible_answers = '\n'.join(item['possible_answers'])
        
        prompt_input = get_template(task='inference', dataset='commonsense_qa', question=fol_question, possible_answers=possible_answers)
        model_output = query_inference_model(prompt_input, model=model, tokenizer=tokenizer)
        item['model_output'] = model_output
        
    with open(f'outputs/{output_filename}.json', 'w') as f:
        json.dump(cqa, f, indent=4)


def hotpotqa(input_filename, output_filename):
    with open(f'outputs/{input_filename}.json', 'r') as f:
        hqa = json.load(f)
    model, tokenizer = load_inference_model()
    for i, item in enumerate(tqdm(hqa)):
        nl_question = item['nl_question']
        fol_question = item['fol_question']

        # nl_contexts = '\n'.join(item['nl_contexts'])
        fol_contexts = '\n'.join(item['fol_contexts'])

        prompt_input = get_template(task='inference', dataset='hotpotqa', nl_question=nl_question, fol_question=fol_question, contexts=fol_contexts)
        model_output = query_inference_model(prompt_input, model=model, tokenizer=tokenizer)
        item['model_output'] = model_output
    
    with open(f'outputs/{output_filename}.json', 'w') as f:
        json.dump(hqa, f, indent=4)    


@click.command()
@click.option('--dataset', '-d', type=str, help='Dataset to perform inference on')
@click.option('--input', '-i', type=str, help='Dataset to perform inference on')
@click.option('--output', '-o', type=str, help='Output filename')
def inference(dataset, input, output):
    if dataset == 'commonsense_qa':
        commonsense_qa(input, output)
    elif dataset == 'hotpotqa':
        hotpotqa(input, output)
    else:
        raise ValueError('Dataset not found')


if __name__ == '__main__':
    inference()
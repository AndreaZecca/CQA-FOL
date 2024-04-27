import click
from datasets import load_dataset
from utils import get_translation_template
import json

def commonsense_qa(split):
    cqa = load_dataset('tau/commonsense_qa', split=split)
    json_cqa = []
    for item in cqa:
        question = item['question'].strip()
        concept = item['question_concept']
        choices = list(zip(item['choices']['label'], item['choices']['text']))
        answer = item['answerKey'] if item['answerKey'].strip() != '' else 'N/A'
        promtp_input = get_translation_template(question)
        fol_question = '' # TODO: Implement LogicLlama's FOL translation
        json_cqa.append({
            'question': question,
            'concept': concept,
            'choices': [f'{label}) {text}' for label, text in choices],
            'answer': answer,
            'fol_question': fol_question
        })
    with open(f'FOL/cqa.json', 'w') as f:
        json.dump(json_cqa, f, indent=4)

def hotpotqa(split):
    hqa = load_dataset('hotpot_qa', 'fullwiki', split=split)
    json_hqa = []
    for item in hqa:
        # TODO: How to deal with contexts

@click.command()
@click.option('--dataset', '-d', help='Dataset to be used', type=click.Choice(['commonsense_qa', 'hotpotqa']))
@click.option('--split', '-s', help='Split to be used', type=click.Choice(['train', 'validation', 'test']), default='train')
def translate(dataset, split):
    if dataset == 'commonsense_qa':
        commonsense_qa(split)
    elif dataset == 'hotpotqa':
        hotpotqa(split)
    else:
        raise ValueError('Dataset not found')

if __name__ == '__main__':
    translate()
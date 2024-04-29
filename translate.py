import click
from datasets import load_dataset
from utils import get_translation_template, load_model, query_model
import json

def commonsense_qa(split):
    cqa = load_dataset('tau/commonsense_qa', split=split)
    model, tokenizer = load_model()
    json_cqa = []
    for item in cqa:
        question = item['question'].strip()
        concept = item['question_concept']
        choices = list(zip(item['choices']['label'], item['choices']['text']))
        answer = item['answerKey'] if item['answerKey'].strip() != '' else 'N/A'
        promtp_input = get_translation_template(dataset='commonsense_qa', question=question)
        fol_question = query_model(question, dataset='commonsense_qa')
        json_cqa.append({
            'question': question,
            'concept': concept,
            'choices': [f'{label}) {text}' for label, text in choices],
            'answer': answer,
            'fol_question': fol_question,
            'prompt_input': promtp_input,
            'fol_question': fol_question
        })
    with open(f'FOL/cqa_{split}.json', 'w') as f:
        json.dump(json_cqa, f, indent=4)

def hotpotqa(split):
    hqa = load_dataset('hotpot_qa', 'fullwiki', split=split)
    model, tokenizer = load_model()
    json_hqa = []
    for item in hqa:
        question = item['question'].strip()
        concept = item['type'].strip()
        answer = item['answer'].strip()
        contexts = []
        for supporing_title, supporting_id in zip(item['supporting_facts']['title'], item['supporting_facts']['sent_id']):
            try:
                sentences_index = item['context']['title'].index(supporing_title)
                sentences = item['context']['sentences'][sentences_index]
                supported_sentence = sentences[supporting_id]
                contexts.append(supported_sentence)
            except:
                continue
        promtp_input = get_translation_template(dataset='hotpotqa', question=question, context="\n".join(contexts))
        fol_question = query_model(question, dataset='hotpotqa')
        json_hqa.append({
            'question': question,
            'concept': concept,
            'answer': answer,
            'contexts': contexts,
            'fol_question': fol_question,
            'prompt_input': promtp_input
        })
    with open(f'FOL/hqa_{split}.json', 'w') as f:
        json.dump(json_hqa, f, indent=4)


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
import click
from datasets import load_dataset
from utils import get_template, load_translation_model, query_translation_model
import json
from tqdm import tqdm

def commonsense_qa(split, output_filename):
    cqa = load_dataset('tau/commonsense_qa', split=split)
    model, tokenizer = load_translation_model()
    json_cqa = []
    for i, item in enumerate(tqdm(cqa)):
        nl_question = item['question'].strip()

        concept = item['question_concept']
        
        zipped_choices = list(zip(item['choices']['label'], item['choices']['text']))
        possible_answers = [f'{label}) {text}' for label, text in zipped_choices]
        
        correct_answer = item['answerKey'] if item['answerKey'].strip() != '' else 'N/A'
        
        question_prompt_input = get_template(task='translation', dataset='commonsense_qa', question=nl_question)
        
        fol_question = query_translation_model(question_prompt_input, model=model, tokenizer=tokenizer)
        
        json_cqa.append({
            'nl_question': nl_question,
            'concept': concept,
            'possible_answers': possible_answers,
            'correct_answer': correct_answer,
            'fol_question': fol_question
        })
    with open(f'outputs/{output_filename}.json', 'w') as f:
        json.dump(json_cqa, f, indent=4)

def hotpotqa(split, output_filename):
    hqa = load_dataset('hotpot_qa', 'fullwiki', split=split)
    model, tokenizer = load_translation_model()
    json_hqa = []
    for i, item in enumerate(tqdm(hqa)):
        if i == 1000:
            break
        nl_question = item['question'].strip()
        
        concept = item['type'].strip()
        
        correct_answer = item['answer'].strip()
        
        nl_contexts = [] 
        for supporing_title, supporting_id in zip(item['supporting_facts']['title'], item['supporting_facts']['sent_id']):
            try:
                sentences_index = item['context']['title'].index(supporing_title)
                sentences = item['context']['sentences'][sentences_index]
                supported_sentence = sentences[supporting_id]
                nl_contexts.append(supported_sentence)
            except:
                continue
            
        fol_contexts = []
        for context in nl_contexts:
            context_prompt_input = get_template(task='translation', dataset='hotpotqa', question=context)
            fol_context = query_translation_model(context_prompt_input, model=model, tokenizer=tokenizer)
            fol_contexts.append(fol_context)

        question_prompt_input = get_template(task='translation', dataset='hotpotqa', question=nl_question)
        fol_question = query_translation_model(question_prompt_input, model=model, tokenizer=tokenizer)
        
        json_hqa.append({
            'nl_question': nl_question,
            'concept': concept,
            'correct_answer': correct_answer,
            'nl_contexts': nl_contexts,
            'fol_question': fol_question,
            'fol_contexts': fol_contexts
        })
    with open(f'outputs/{output_filename}.json', 'w') as f:
        json.dump(json_hqa, f, indent=4)


@click.command()
@click.option('--dataset', '-d', help='Dataset to be used', type=click.Choice(['commonsense_qa', 'hotpotqa']))
@click.option('--split', '-s', help='Split to be used', type=click.Choice(['train', 'validation', 'test']), default='train')
@click.option('--output', '-o', help='Output file name', required=True)
def translate(dataset, split, output):
    if dataset == 'commonsense_qa':
        commonsense_qa(split, output)
    elif dataset == 'hotpotqa':
        hotpotqa(split, output)
    else:
        raise ValueError('Dataset not found')

if __name__ == '__main__':
    translate()

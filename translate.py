import click
from datasets import load_dataset
from utils import get_template, query_translation_model, load_vllm_translation_model, query_vllm_model
import json
from tqdm import tqdm

def commonsense_qa(split, template_filename, output_filename, BATCH_SIZE = 8):
    cqa = load_dataset('tau/commonsense_qa', split=split)
    model = load_vllm_translation_model()
    json_cqa = []
    iterations = len(cqa) // BATCH_SIZE
    for i in tqdm(range(iterations + 1)):
        b_start = i * BATCH_SIZE
        b_end = min((i + 1) * BATCH_SIZE, len(cqa))
        batch = [cqa[j] for j in range(b_start, b_end)]

        batch_questions = [item['question'] for item in batch]
        batch_concepts = [item['question_concept'] for item in batch]
        batch_choices = [list(zip(item['choices']['label'], item['choices']['text'])) for item in batch]
        batch_possible_answers = [[f'{label}) {text}' for label, text in choices] for choices in batch_choices]
        batch_correct_answers = [item['answerKey'] if item['answerKey'].strip() != '' else 'N/A' for item in batch]

        batch_prompt_inputs = [get_template(task='translation', template_filename=template_filename, formatting_kwargs={'question': question}) for question in batch_questions]
        
        batch_fol_questions = query_vllm_model(model=model, input_prompts=batch_prompt_inputs)
        
        for j in range(len(batch)):
            json_cqa.append({
                'nl_question': batch_questions[j],
                'concept': batch_concepts[j],
                'possible_answers': batch_possible_answers[j],
                'correct_answer': batch_correct_answers[j],
                'fol_question': batch_fol_questions[j]
            })            
    with open(f'outputs/{output_filename}.json', 'w', encoding='utf-8') as f:
        json.dump(json_cqa, f, indent=4, ensure_ascii=False)


def hotpotqa(split, template_filename, output_filename, BATCH_SIZE = 8):
    hqa = load_dataset('hotpot_qa', 'fullwiki', split=split)
    model = load_vllm_translation_model()
    json_hqa = []
    for item in enumerate(tqdm(hqa)):
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
            
        
        question_template = get_template(task='translation', template_filename=template_filename, formatting_kwargs={'question': nl_question}),
        contexts_template = [get_template(task='translation', template_filename=template_filename, formatting_kwargs={'question': context}) for context in nl_contexts]
        prompt_inputs = [
            *question_template,
            *contexts_template
        ]

        fol_results = query_vllm_model(model=model, input_prompts=prompt_inputs, max_tokens=512)

        fol_question = fol_results[0]
        fol_contexts = fol_results[1:]
        
        json_hqa.append({
            'nl_question': nl_question,
            'concept': concept,
            'correct_answer': correct_answer,
            'nl_contexts': nl_contexts,
            'fol_question': fol_question,
            'fol_contexts': fol_contexts
        })

    with open(f'outputs/{output_filename}.json', 'w', encoding='utf-8') as f:
        json.dump(json_hqa, f, indent=4, ensure_ascii=False)


@click.command()
@click.option('--dataset', '-d', help='Dataset to be used', type=click.Choice(['cqa', 'hqa']))
@click.option('--split', '-s', help='Split to be used', type=click.Choice(['train', 'validation', 'test']), default='train')
@click.option('--template', '-t', help='Template file name', required=True)
@click.option('--output', '-o', help='Output file name', required=True)
@click.option('--batch_size', '-b', help='Batch size', default=8, required=False, type=int)
def translate(dataset, split, template, output, batch_size):
    if dataset == 'cqa':
        commonsense_qa(split, template, output, batch_size)
    elif dataset == 'hqa':
        hotpotqa(split, template, output, batch_size)
    else:
        raise ValueError('Dataset not found')

if __name__ == '__main__':
    translate()

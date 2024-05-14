import click
from datasets import load_dataset
from utils import get_template, load_vllm_translation_model, query_vllm_model
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

        batch_prompt_inputs = [get_template(task='translation', dataset='cqa', template_filename=template_filename, formatting_kwargs={'question': question}) for question in batch_questions]
        
        batch_fol_questions = query_vllm_model(model=model, input_prompts=batch_prompt_inputs)
        
        for j in range(len(batch)):
            json_cqa.append({
                'nl_question': batch_questions[j],
                'concept': batch_concepts[j],
                'possible_answers': batch_possible_answers[j],
                'correct_answer': batch_correct_answers[j],
                'fol_question': batch_fol_questions[j]
            })            
    with open(f'outputs/cqa/{output_filename}', 'w', encoding='utf-8') as f:
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
            
        
        question_template = get_template(task='translation', dataset='hqa', template_filename=template_filename, formatting_kwargs={'question': nl_question}),
        contexts_template = [get_template(task='translation', dataset='hqa', template_filename=template_filename, formatting_kwargs={'question': context}) for context in nl_contexts]
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

    with open(f'outputs/hqa/{output_filename}', 'w', encoding='utf-8') as f:
        json.dump(json_hqa, f, indent=4, ensure_ascii=False)

def openbookqa(split, template_filename, output_filename, BATCH_SIZE = 8):
    obqa = load_dataset('allenai/openbookqa', 'additional', split=split)
    model = load_vllm_translation_model()
    json_obqa = []
    iterations = len(obqa) // BATCH_SIZE
    for i in tqdm(range(iterations + 1)):
        b_start = i * BATCH_SIZE
        b_end = min((i + 1) * BATCH_SIZE, len(obqa))
        batch = [obqa[j] for j in range(b_start, b_end)]

        batch_questions = [item['question_stem'] for item in batch]
        batch_choices = [list(zip(item['choices']['label'],item['choices']['text'])) for item in batch]
        batch_possible_answers = [[f'{label}) {text}' for label, text in choices] for choices in batch_choices]
        batch_correct_answers = [item['answerKey'] if item['answerKey'].strip() != '' else 'N/A' for item in batch]
        batch_contexts = [item['fact1'] for item in batch]

        questions_templates = [get_template(task='translation', dataset='obqa', template_filename=template_filename, formatting_kwargs={'question': question}) for question in batch_questions]
        contexts_templates = [get_template(task='translation', dataset='obqa', template_filename=template_filename, formatting_kwargs={'question': context}) for context in batch_contexts]

        prompt_inputs = [
            *questions_templates,
            *contexts_templates
        ]

        fol_results = query_vllm_model(model=model, input_prompts=prompt_inputs, max_tokens=512)

        fol_questions = fol_results[:len(batch)]
        fol_contexts = fol_results[len(batch):]

        for j in range(len(batch)):
            json_obqa.append({
                'nl_question': batch_questions[j],
                'nl_context': batch_contexts[j] if len(batch_contexts) > 0 else 'N/A',
                'possible_answers': batch_possible_answers[j],
                'correct_answer': batch_correct_answers[j],
                'fol_question': fol_questions[j],
                'fol_context': fol_contexts[j] if len(fol_contexts) > 0 else 'N/A'
            })

    with open(f'outputs/obqa/{output_filename}', 'w', encoding='utf-8') as f:
        json.dump(json_obqa, f, indent=4, ensure_ascii=False)

def cqa20(split, template_filename, output_filename, BATCH_SIZE = 8):
    cqa20 = load_dataset('tasksource/commonsense_qa_2.0', split=split)
    model = load_vllm_translation_model()
    json_cqa_2_0 = []
    iterations = len(cqa20) // BATCH_SIZE
    for i in tqdm(range(iterations+1)):
        b_start = i * BATCH_SIZE
        b_end = min((i + 1) * BATCH_SIZE, len(cqa20))
        batch = [cqa20[j] for j in range(b_start, b_end)]

        batch_questions = [item['question'] for item in batch]
        batch_correct_answers = [item['answer'] if item['answer'].strip() != '' else 'N/A' for item in batch]

        batch_prompt_inputs = [get_template(task='translation', dataset='cqa20', template_filename=template_filename, formatting_kwargs={'question': question}) for question in batch_questions]

        batch_fol_questions = query_vllm_model(model=model, input_prompts=batch_prompt_inputs)

        for j in range(len(batch)):
            json_cqa_2_0.append({
                'nl_question': batch_questions[j],
                'correct_answer': batch_correct_answers[j],
                'fol_question': batch_fol_questions[j]
            })

    with open(f'outputs/cqa20/{output_filename}', 'w', encoding='utf-8') as f:
        json.dump(json_cqa_2_0, f, indent=4, ensure_ascii=False)

def snli(split, template_filename, output_filename, BATCH_SIZE=8):
    label_2_nl = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    }
    nllabel_2_abc = {
        'entailment': 'A',
        'neutral': 'B',
        'contradiction': 'C'
    }
    possible_answers = 'A) entailment\n B) neutral\n C) contradiction'
    snli = load_dataset('snli', split=split)
    snli = [i for i in snli if i['label'] != -1]
    model = load_vllm_translation_model()
    json_snli = []
    iterations = len(snli) // BATCH_SIZE
    for i in tqdm(range(iterations + 1)):
        b_start = i * BATCH_SIZE
        b_end = min((i + 1) * BATCH_SIZE, len(snli))
        batch = [snli[j] for j in range(b_start, b_end)]

        batch_premise = [item['premise'] for item in batch]
        batch_hypothesis = [item['hypothesis'] for item in batch]

        batch_labels = [label_2_nl[item['label']] for item in batch]
        batch_correct_answers = [nllabel_2_abc[label] for label in batch_labels]


        batch_prompt_inputs_premises = [get_template(task='translation', dataset='snli', template_filename=template_filename, formatting_kwargs={'question': premise}) for premise in batch_premise]
        batch_prompt_inputs_hypothesis = [get_template(task='translation', dataset='snli', template_filename=template_filename, formatting_kwargs={'question': hypothesis}) for hypothesis in batch_hypothesis]

        batch_fol_premises = query_vllm_model(model=model, input_prompts=batch_prompt_inputs_premises)
        batch_fol_hypothesis = query_vllm_model(model=model, input_prompts=batch_prompt_inputs_hypothesis)

        for j in range(len(batch)):
            json_snli.append({
                'nl_premise': batch_premise[j],
                'nl_hypothesis': batch_hypothesis[j],
                'possible_answers': possible_answers,
                'correct_answer': batch_correct_answers[j],
                'fol_premise': batch_fol_premises[j],
                'fol_hypothesis': batch_fol_hypothesis[j]
            })

    with open(f'outputs/snli/{output_filename}', 'w', encoding='utf-8') as f:
        json.dump(json_snli, f, indent=4, ensure_ascii=False)

@click.command()
@click.option('--dataset', '-d', help='Dataset to be used', type=click.Choice(['cqa', 'hqa', 'obqa', 'cqa20', 'snli']))
@click.option('--split', '-s', help='Split to be used', type=click.Choice(['train', 'validation', 'test']), default='train')
@click.option('--template', '-t', help='Template file name', required=True)
@click.option('--output', '-o', help='Output file name', required=True)
@click.option('--batch_size', '-b', help='Batch size', default=8, required=False, type=int)
def translate(dataset, split, template, output, batch_size):
    callables = {
        'cqa': commonsense_qa,
        'hqa': hotpotqa,
        'obqa': openbookqa,
        'cqa20': cqa20,
        'snli': snli
    }
    if dataset in callables:
        callables[dataset](split, template, output, batch_size)    
    else:
        raise ValueError('Dataset not found')

if __name__ == '__main__':
    translate()

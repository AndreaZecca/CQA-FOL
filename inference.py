import click
from utils import get_template, load_vllm_inference_model, query_vllm_model
import json
from tqdm import tqdm

def commonsense_qa(input_filename, output_filename, template_filename, BATCH_SIZE=8):
    with open(f'outputs/cqa/{input_filename}', 'r', encoding='utf-8') as f:
        cqa = json.load(f)
    model = load_vllm_inference_model()
    iterations = len(cqa) // BATCH_SIZE
    for i in tqdm(range(iterations+1)):
        b_start = i * BATCH_SIZE
        b_end = min((i+1) * BATCH_SIZE, len(cqa))
        if b_start > 10:
            break
        batch = cqa[b_start:b_end]

        nl_questions = [item['nl_question'] for item in batch]
        fol_questions = [item['fol_question'] for item in batch]

        possible_answers = ['\n'.join(item['possible_answers']) for item in batch]

        input_prompts = [get_template(task='inference', dataset='cqa', template_filename=template_filename, formatting_kwargs={
            'nl_question': nl_question,
            'fol_question': fol_question,
            'possible_answers': possible_answer
        }) for nl_question, fol_question, possible_answer in zip(nl_questions, fol_questions, possible_answers)]

        model_outputs = query_vllm_model(model=model, input_prompts=input_prompts, max_tokens=8)
        for j, item in enumerate(batch):
            model_output = model_outputs[j]
            model_answer = model_output.split('### Answer')[-1].strip()
            item['model_output'] = model_output
            item['model_answer'] = model_answer

    with open(f'outputs/cqa/{output_filename}', 'w',  encoding='utf-8') as f:
        json.dump(cqa, f, indent=4, ensure_ascii=False)

def hotpotqa(input_filename, output_filename, template_filename, BATCH_SIZE=8):
    with open(f'outputs/hqa/{input_filename}', 'r', encoding='utf-8') as f:
        hqa = json.load(f)
    model = load_vllm_inference_model()
    iterations = len(hqa) // BATCH_SIZE
    for i in tqdm(range(iterations+1)):
        b_start = i * BATCH_SIZE
        b_end = min((i+1) * BATCH_SIZE, len(hqa))
        batch = hqa[b_start:b_end]
        
        nl_questions = [item['nl_question'] for item in batch]
        fol_questions = [item['fol_question'] for item in batch]

        nl_contexts = ['\n'.join(item['nl_contexts']) for item in batch]
        fol_contexts = ['\n'.join(item['fol_contexts']) for item in batch]

        input_prompts = [get_template(task='inference', dataset='hqa', template_filename=template_filename, formatting_kwargs={
            'nl_contexts': nl_context,
            # 'fol_contexts': fol_context,
            'nl_question': nl_question,
            # 'fol_question': fol_question
        }) for nl_context, fol_context, nl_question, fol_question in zip(nl_contexts, fol_contexts, nl_questions, fol_questions)]

        model_outputs = query_vllm_model(model=model, input_prompts=input_prompts, max_tokens=8)

        for j, item in enumerate(batch):
            model_output = model_outputs[j]
            model_answer = model_output.split('### Answer')[-1].strip()
            item['model_output'] = model_output
            item['model_answer'] = model_answer

    with open(f'outputs/hqa/{output_filename}', 'w', encoding='utf-8') as f:
        json.dump(hqa, f, indent=4, ensure_ascii=False)

def openbookqa(input_filename, output_filename, template_filename, BATCH_SIZE=8):
    with open(f'outputs/obqa/{input_filename}', 'r', encoding='utf-8') as f:
        obqa = json.load(f)

    model = load_vllm_inference_model()
    iterations = len(obqa) // BATCH_SIZE
    for i in tqdm(range(iterations+1)):
        b_start = i * BATCH_SIZE
        b_end = min((i+1) * BATCH_SIZE, len(obqa))
        batch = obqa[b_start:b_end]

        nl_questions = [item['nl_question'] for item in batch]
        fol_questions = [item['fol_question'] for item in batch]

        nl_contexts = [item['nl_context'] for item in batch]
        fol_contexts = [item['fol_context'] for item in batch]

        possible_answers = ['\n'.join(item['possible_answers']) for item in batch]

        input_prompts = [get_template(task='inference', dataset='obqa', template_filename=template_filename, formatting_kwargs={
            # 'nl_context': nl_context,
            'fol_context': fol_context,
            # 'nl_question': nl_question,
            'fol_question': fol_question,
            'possible_answers': possible_answer
        }) for nl_context, fol_context, nl_question, fol_question, possible_answer in zip(nl_contexts, fol_contexts, nl_questions, fol_questions, possible_answers)]

        model_outputs = query_vllm_model(model=model, input_prompts=input_prompts, max_tokens=8)

        for j, item in enumerate(batch):
            model_output = model_outputs[j]
            model_answer = model_output.split('### Answer')[-1].strip()
            item['model_output'] = model_output
            item['model_answer'] = model_answer

    with open(f'outputs/obqa/{output_filename}', 'w', encoding='utf-8') as f:
        json.dump(obqa, f, indent=4, ensure_ascii=False)

def cqa20(input_filename, output_filename, template_filename, BATCH_SIZE=8):
    with open(f'outputs/cqa20/{input_filename}', 'r', encoding='utf-8') as f:
        cqa20 = json.load(f)

    model = load_vllm_inference_model()
    iterations = len(cqa20) // BATCH_SIZE
    for i in tqdm(range(iterations+1)):
        b_start = i * BATCH_SIZE
        b_end = min((i+1) * BATCH_SIZE, len(cqa20))
        batch = cqa20[b_start:b_end]

        nl_questions = [item['nl_question'] for item in batch]
        fol_questions = [item['fol_question'] for item in batch]

        correct_answers = [item['correct_answer'] for item in batch]
        possible_answers = ['A) yes\nB) no' for item in batch]

        input_prompts = [get_template(task='inference', dataset='cqa20', template_filename=template_filename, formatting_kwargs={
            # 'nl_question': nl_question,
            'fol_question': fol_question,
            'possible_answers': possible_answer
        }) for nl_question, fol_question, possible_answer in zip(nl_questions, fol_questions, possible_answers)]

        model_outputs = query_vllm_model(model=model, input_prompts=input_prompts, max_tokens=8)

        for j, item in enumerate(batch):
            model_output = model_outputs[j]
            model_answer = model_output.split('### Answer')[-1].strip()
            item['model_output'] = model_output
            item['model_answer'] = model_answer

    with open(f'outputs/cqa20/{output_filename}', 'w', encoding='utf-8') as f:
        json.dump(cqa20, f, indent=4, ensure_ascii=False)

def snli(input_filename, output_filename, template_filename, BATCH_SIZE=8):
    with open(f'outputs/snli/{input_filename}', 'r', encoding='utf-8') as f:
        snli = json.load(f)
    model = load_vllm_inference_model()
    iterations = len(snli) // BATCH_SIZE
    for i in tqdm(range(iterations+1)):
        b_start = i * BATCH_SIZE
        b_end = min((i+1) * BATCH_SIZE, len(snli))
        batch = snli[b_start:b_end]

        nl_premises = [item['nl_premise'] for item in batch]
        nl_hypotheses = [item['nl_hypothesis'] for item in batch]

        fol_premises = [item['fol_premise'] for item in batch]
        fol_hypotheses = [item['fol_hypothesis'] for item in batch]

        possible_answers = [item['possible_answers'] for item in batch]
        correct_answers = [item['correct_answer'] for item in batch]

        input_prompts = [get_template(task='inference', dataset='snli', template_filename=template_filename, formatting_kwargs={
            'nl_premise': nl_premise,
            # 'fol_premise': fol_premise,
            'nl_hypothesis': nl_hypothesis,
            # 'fol_hypothesis': fol_hypothesis,
            'possible_answers': possible_answer
        }) for nl_premise, nl_hypothesis, fol_premise, fol_hypothesis, possible_answer in zip(nl_premises, nl_hypotheses, fol_premises, fol_hypotheses, possible_answers)]

        model_outputs = query_vllm_model(model=model, input_prompts=input_prompts, max_tokens=8)

        for j, item in enumerate(batch):
            model_output = model_outputs[j]
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
@click.option('--batch', '-b', type=int, help='Batch size for inference', default=8)
def inference(dataset, input, output, template, batch):
    callables = {
        'cqa': commonsense_qa,
        'hqa': hotpotqa,
        'obqa': openbookqa,
        'cqa20': cqa20,
        'snli': snli
    }

    if dataset in callables:
        callables[dataset](input, output, template, batch)    
    else:
        raise ValueError('Dataset not found')


if __name__ == '__main__':
    inference()
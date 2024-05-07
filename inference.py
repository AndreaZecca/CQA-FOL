import click
from utils import get_template, load_inference_model, query_model
import json
from tqdm import tqdm

def commonsense_qa(input_filename, output_filename, template_filename):
    with open(f'outputs/{input_filename}.json', 'r', encoding='utf-8') as f:
        cqa = json.load(f)
    model, tokenizer = load_inference_model()
    for i, item in enumerate(tqdm(cqa)):
        nl_question = item['nl_question']
        fol_question = item['fol_question']

        possible_answers = '\n'.join(item['possible_answers'])
        
        input_prompt = get_template(task='inference', template_filename=template_filename, formatting_kwargs={
            'nl_question': nl_question,
            'fol_question': fol_question,
            'possible_answers': possible_answers
        })
        
        model_output = query_model(tokenizer=tokenizer, model=model, input_prompt=input_prompt, max_tokens=8)
        model_answer = model_output.split('### Answer')[-1].strip()
        item['model_output'] = model_output
        item['model_answer'] = model_answer
        
    with open(f'outputs/{output_filename}.json', 'w',  encoding='utf-8') as f:
        json.dump(cqa, f, indent=4, ensure_ascii=False)


def hotpotqa(input_filename, output_filename, template_filename):
    with open(f'outputs/{input_filename}.json', 'r', encoding='utf-8') as f:
        hqa = json.load(f)
    model, tokenizer = load_inference_model()
    for i, item in enumerate(tqdm(hqa)):
        nl_question = item['nl_question']
        fol_question = item['fol_question']

        nl_contexts = '\n'.join(item['nl_contexts'])
        fol_contexts = '\n'.join(item['fol_contexts'])
        
        input_prompt = get_template(task='inference', template_filename=template_filename, formatting_kwargs={
            'fol_contexts': fol_contexts,
            'fol_question': fol_question,
        })
        
        model_output = query_model(tokenizer=tokenizer, model=model, input_prompt=input_prompt, max_tokens=8)
        model_answer = model_output.split('### Answer')[-1].strip()
        item['model_output'] = model_output
        item['model_answer'] = model_answer

    with open(f'outputs/{output_filename}.json', 'w', encoding='utf-8') as f:
        json.dump(hqa, f, indent=4, ensure_ascii=False)


# def commonsense_qa(input_filename, output_filename, template_filename, BATCH_SIZE=8):
#     with open(f'outputs/{input_filename}.json', 'r') as f:
#         cqa = json.load(f)
#     model = load_vllm_inference_model()
#     iterations = len(cqa) // BATCH_SIZE
#     for i in tqdm(range(iterations + 1)):
#         b_start = i * BATCH_SIZE
#         b_end = min((i + 1) * BATCH_SIZE, len(cqa))
#         batch = [cqa[j] for j in range(b_start, b_end)]

#         batch_nl_questions = [item['nl_question'] for item in batch]
#         batch_fol_questions = [item['fol_question'] for item in batch]

#         batch_possible_answers = ['\n'.join(item['possible_answers']) for item in batch]
        
#         batch_prompt_inputs = [
#             get_template(task='inference', template_filename=template_filename, formatting_kwargs={
#                 'fol_question': fol_question,
#                 'possible_answers': possible_answers
#             }) for fol_question, possible_answers in zip(batch_fol_questions, batch_possible_answers)
#         ]

#         batch_model_outputs = query_vllm_model(model=model, input_prompts=batch_prompt_inputs, max_tokens=8)
#         for j, item in enumerate(batch):
#             item['model_output'] = batch_model_outputs[j]

#     with open(f'outputs/{output_filename}.json', 'w',  encoding='utf-8') as f:
#         json.dump(cqa, f, indent=4, ensure_ascii=False)



# def hotpotqa(input_filename, output_filename, template_filename, BATCH_SIZE=8):
#     with open(f'outputs/{input_filename}.json', 'r') as f:
#         hqa = json.load(f)
#     model = load_vllm_inference_model()
#     iterations = len(hqa) // BATCH_SIZE
#     for i in tqdm(range(iterations + 1)):
#         b_start = i * BATCH_SIZE
#         b_end = min((i + 1) * BATCH_SIZE, len(hqa))
#         batch = [hqa[j] for j in range(b_start, b_end)]

#         batch_nl_questions = [item['nl_question'] for item in batch]
#         batch_fol_questions = [item['fol_question'] for item in batch]

#         batch_nl_contexts = ['\n'.join(item['nl_contexts']) for item in batch]
#         batch_fol_contexts = ['\n'.join(item['fol_contexts']) for item in batch]

#         batch_prompt_inputs = [
#             #TODO: Fix this
#         ]

#         batch_model_outputs = query_vllm_model(model=model, input_prompts=batch_prompt_inputs, max_tokens=8)

#         for j, item in enumerate(batch):
#             item['model_output'] = batch_model_outputs[j]

#     with open(f'outputs/{output_filename}.json', 'w',  encoding='utf-8') as f:
#         json.dump(hqa, f, indent=4, ensure_ascii=False)


@click.command()
@click.option('--dataset', '-d', help='Dataset to be used', type=click.Choice(['cqa', 'hqa']))
@click.option('--input', '-i', type=str, help='Dataset to perform inference on')
@click.option('--output', '-o', type=str, help='Output filename')
@click.option('--template', '-t', type=str, help='Template filename for inference prompt')
# @click.option('--batch_size', '-b', type=int, default=8, help='Batch size for inference')
def inference(dataset, input, output, template):
    if dataset == 'cqa':
        commonsense_qa(input, output, template)
    elif dataset == 'hqa':
        hotpotqa(input, output, template)
    else:
        raise ValueError('Dataset not found')


if __name__ == '__main__':
    inference()
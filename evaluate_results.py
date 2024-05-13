import json
import os
import evaluate
from tqdm import tqdm

skip_files = [
    'cqa_val.json',
    'hqa_val.json',
    'obqa_test.json',
    'cqa20.json'
]

def eval_mcqa(data):
    count = 0
    for item in data:
        correct_answer = item['correct_answer']
        model_answer = item['model_answer']
        model_answer = model_answer if len(model_answer) == 1 else 'N/A'
        if correct_answer.strip().lower() == model_answer.strip().lower():
            count += 1
    return round(count / len(data), 2)

def eval_rouge(data):
    rouge = evaluate.load('rouge')
    predictions = [item['model_answer'] for item in data][:3]
    references = [item['correct_answer'] for item in data][:3]
    scores = rouge.compute(predictions=predictions, references=references)
    return round(scores['rougeLsum'], 2)

def evaluate_results():
    results = {}
    for result_filename in tqdm(sorted(os.listdir('outputs'))):
        if not result_filename.endswith('.json') or result_filename in skip_files:
            continue
        with open(f'outputs/{result_filename}', 'r') as f:
            data = json.load(f)
        if result_filename.split('_')[0] in ['cqa', 'obqa', 'cqa20']:
            res = eval_mcqa(data)
        elif result_filename.split('_')[0] in ['hqa']:
            res = eval_rouge(data)
        else:
            print(f'Unknown dataset {result_filename.split("_")[0]}')
            continue
        results[result_filename] = res

    print('Results:')
    for result_filename, acc in results.items():
        print(f'{result_filename}: {acc}')


if __name__ == '__main__':
    evaluate_results()
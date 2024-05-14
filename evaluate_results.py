import json
import os
import evaluate
from tqdm import tqdm

skip_files = [
    'train.json',
    'validation.json',
    'test.json'
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
    return scores

def evaluate_results():
    results = {}
    for dataset in tqdm(os.listdir('outputs')):
        if dataset not in ['cqa', 'cqa20', 'hqa', 'obqa', 'snli']:
            continue
        results[dataset] = {}
        for result_filename in os.listdir(f'outputs/{dataset}'):
            if not result_filename.endswith('.json') or result_filename in skip_files:
                continue
            with open(f'outputs/{dataset}/{result_filename}', 'r', encoding='utf-8') as f:
                data = json.load(f)
            if dataset in ['hqa']:
                res = eval_rouge(data)
            elif dataset in ['cqa', 'cqa20', 'obqa', 'snli']:
                res = eval_mcqa(data)
            else:
                print(f'Unknown dataset {result_filename.split("_")[0]}')
                continue
            results[dataset][result_filename.split('.json')[0]] = res
    for dataset, results_dataset in results.items():
        print(f'\nDataset: {dataset}')
        for result_filename, result in results_dataset.items():
            if isinstance(result, dict):
                print(f'    {result_filename}:')
                for metric, value in result.items():
                    print(f'        {metric}: {value}')
            else:
                print(f'    {result_filename}: {result}')               

if __name__ == '__main__':
    evaluate_results()
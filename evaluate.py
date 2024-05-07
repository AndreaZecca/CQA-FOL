import json
import os
from tqdm import tqdm

skip_files = [
    'cqa_val.json',
    'hqa_val.json',
]

def eval_cqa(data):
    count = 0
    for item in tqdm(data):
        correct_answer = item['correct_answer']
        model_answer = item['model_answer']
        model_answer = model_answer if len(model_answer) == 1 else 'N/A'
        if correct_answer.strip().lower() == model_answer.strip().lower():
            count += 1
    return round(count / len(data), 2)

def eval_hqa(data):
    pass

def evaluate():
    results = {}
    for result_filename in sorted(os.listdir('outputs')):
        if not result_filename.endswith('.json') or result_filename in skip_files:
            continue
        with open(f'outputs/{result_filename}', 'r') as f:
            data = json.load(f)
        if 'cqa' in result_filename:
            acc = eval_cqa(data)
        elif 'hqa' in result_filename:
            acc = eval_hqa(data)
        else:
            print('Unknown dataset')
            continue
        results[result_filename] = acc

    print('Results:')
    for result_filename, acc in results.items():
        print(f'{result_filename}: {acc}')

if __name__ == '__main__':
    evaluate()
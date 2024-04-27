from datasets import load_dataset
import json
from tqdm import tqdm

def load_from_huggingface(dataset_name):
    dataset = load_dataset(dataset_name)
    train, val, test = dataset['train'], dataset['validation'], dataset['test']
    return train, val, test

def convert_to_json_cqa(dataset):
    json_data = []
    for row in tqdm(dataset):
        question = row['question'].strip()
        answer = row['answerKey'] if row['answerKey'].strip() != '' else 'N/A'
        choices = [f'{l.strip()}) {t.strip()}' for l,t in zip(row['choices']['label'], row['choices']['text'])]
        json_data.append({'question': question, 'answer': answer, 'choices': choices})
    return json_data

def commonsense_qa():
    train, validation, test = load_from_huggingface('tau/commonsense_qa')
    train = convert_to_json_cqa(train)
    validation = convert_to_json_cqa(validation)
    test = convert_to_json_cqa(test)
    commonsense_qa_dataset = {
        'train': train,
        'validation': validation,
        'test': test
    }
    with open('data/commonsense_qa.json', 'w') as f:
        json.dump(commonsense_qa_dataset, f, indent=4)
    

def main():
    commonsense_qa()


if __name__ == '__main__':
    main()

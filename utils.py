import os

def get_translation_template(dataset, **formatting_kwargs):
    base_template = open(f'./templates/translation/{dataset}.txt', 'r').read()
    template = base_template.format(**formatting_kwargs)
    return template

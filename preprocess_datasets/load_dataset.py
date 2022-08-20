import os, sys
import json
from datasets import load_dataset

for name in ['cola', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb']:

    os.makedirs(os.path.join(os.path.abspath(os.pardir), 'glue_datasets', name), exist_ok=True)
    
    dataset = load_dataset('glue', name)
    print(dataset)

    for k in ['train', 'validation', 'test']:

        out_file = open(f'datasets/{name}/{k}.json', 'w')
        
        for line in dataset[k]:
            line = json.dumps(line)

            out_file.write(f'{line}\n')


for name in ['mnli']:

    os.makedirs(os.path.join(os.path.abspath(os.pardir), 'glue_datasets', name), exist_ok=True)
    
    dataset = load_dataset('glue', name)
    print(dataset)

    for k in ['train', 'validation_matched', 'test_matched', 'validation_mismatched', 'test_mismatched']:

        out_file = open(f'datasets/{name}/{k}.json', 'w')
        
        for line in dataset[k]:
            line = json.dumps(line)

            out_file.write(f'{line}\n')

import os
import json
import numpy as np
from collections import Counter
from get_description import load_dict

def get_all_files(path):
    if os.path.isfile(path): return [path]
    return [f for d in os.listdir(path)
              for f in get_all_files(os.path.join(path, d))]

def replace_special_tokens(s):
    return s.replace('-', ' ').replace('\'s', ' ').replace('/', ' ').replace('\'', ' ').replace('\"', ' ').replace('?', ' ').split()


wiktionary_path = os.path.join(
    os.path.abspath(os.pardir), 'preprocess_wiktionary', 'wiktionary.json')

wiktionary = json.load(open(wiktionary_path, encoding='utf-8'))
print('wiktionary is successfully loaded!')
print('Started!')

ifolder = os.path.join(os.path.abspath(os.pardir), 'glue_datasets')
for task in ['cola', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'mnli']:
    print(task)
    task_files = get_all_files(os.path.join(ifolder, task))

    vocab_path = os.path.join(ifolder, task, 'vocab.txt')
    rare_vocab_path = os.path.join(ifolder, task, 'vocab.rare.txt')
    output_dict_path = os.path.join(ifolder, task, 'vocab.90.json')
    
    vocab_file = open(vocab_path, 'w')
    rare_vocab = open(rare_vocab_path, 'w')

    word_collections = []
    for idx, file in enumerate(task_files):
        if not file.endswith('prc.json'): continue

        with open(file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line)

                for key, value in line.items():
                    if key in ['idx', 'labels', 'label']: continue
                    word_collections += replace_special_tokens(value)
    
    word2freq = Counter(word_collections)

    vocab = sorted(word2freq.items(), key=lambda k: k[1], reverse=True)
    threshold = np.sum(list(word2freq.values())) * 0.90

    for word, count in vocab:
        vocab_file.write('{}\t{}\n'.format(word, count))
        if threshold - count > 0:
            threshold -= count
            continue
        rare_vocab.write('{}\t{}\n'.format(word, count))

    load_dict(rare_vocab_path, output_dict_path, wiktionary)

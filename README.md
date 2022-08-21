
## Dict-BERT: Enhancing Language Model Pre-training with Dictionary

## Introduction

-- This is the pytorch implementation of our [ACL 2022](https://www.2022.aclweb.org/) paper "*Dict-BERT: Enhancing Language Model Pre-training with Dictionary*" [\[PDF\]](https://arxiv.org/abs/2110.06490). 
In this paper, we propose DictBERT, which is a novel pre-trained language model by leveraging rare word definitions in English dictionaries (e.g., Wiktionary). DictBERT is based on the BERT architecture, trained under the same setting as BERT. Please refer more details in our paper.

## Install the packages

python version >=3.6


```
transformers==4.7.0
datasets==1.8.0
torch==1.8.0
```

Also need to install `dataclasses`, `scipy`, `sklearn`, `nltk`

<!-- pip install label-studio --ignore-installed certifi -->

## Preprocess the data

-- download Wiktionary 

```bash
cd preprocess_wiktionary
bash download_wiktionary.sh
```

-- download GLUE benchmark
```bash
cd preprocess_datasets
bash load_preprocess.sh
```

## Download the checkpoint

https://huggingface.co/wyu1/DictBERT

## Run experiments on GLUE

-- without dictionary

```bash
cd finetune_wo_wiktionary
bash finetune.sh
```

-- with dictionary

```bash
cd finetune_wi_wiktionary
bash finetune.sh
```


## Citation

```
@inproceedings{yu2022dict,
  title={Dict-BERT: Enhancing Language Model Pre-training with Dictionary},
  author={Yu, Wenhao and Zhu, Chenguang and Fang, Yuwei and Yu, Donghan and Wang, Shuohang and Xu, Yichong and Zeng, Michael and Jiang, Meng},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2022},
  pages={1907--1918},
  year={2022}
}
```

Please kindly cite our paper if you find this paper and the codes helpful.

## Acknowledgements

Many thanks to the Github repository of [Transformers](https://github.com/huggingface/transformers). Part of our codes are modified based on their codes.

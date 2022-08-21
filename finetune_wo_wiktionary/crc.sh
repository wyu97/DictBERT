#!/bin/bash
#$ -M wyu1@nd.edu
#$ -m abe
#$ -q gpu@qa-2080ti-006
#$ -pe smp 1
#$ -l gpu=0


export device_num=1
# export model_name=bert-base-uncased
export model_name=wyu1/DictBERT
export outfolder=output-test-standard

CUDA_VISIBLE_DEVICES=$device_num /afs/crc.nd.edu/user/w/wyu1/.conda/envs/dpt/bin/python -u finetune.py \
  --model_name_or_path $model_name \
  --task_name $task \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir $outfolder/$task/

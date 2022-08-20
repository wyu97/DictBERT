#!/bin/bash

export task=cola
export outfolder=output-${task}
# export model_name=bert-base-uncased
export model_name=wyu1/DictBERT


if [[ ${task} =~ $'cola' ]] || [[ ${task} =~ $'sst2' ]]
then
  epoch=10; max_length=128
elif [[ ${task} =~ $'qqp' ]] || [[ ${task} =~ $'mnli' ]] || [[ ${task} =~ $'qnli' ]]
then
  epoch=5; max_length=128
elif [[ ${OUTPUT_NAME} =~ $'rte' ]]
then
  epoch=10; max_length=256
elif [[ ${OUTPUT_NAME} =~ $'mrpc' ]] || [[ ${task} =~ $'stsb' ]]
then
  epoch=5; max_length=256
fi


python -u finetune.py \
  --model_name_or_path $model_name \
  --task_name $task \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length $max_length \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs $epoch \
  --output_dir $outfolder/$task/

#!/bin/bash
#$ -M wyu1@nd.edu
#$ -m abe
#$ -q gpu@qa-2080ti-007
#$ -pe smp 1
#$ -l gpu=0

export device_num=2
export model_name=wyu1/DictBERT
export input_dir=/afs/crc.nd.edu/group/dmsquare/vol4/wyu1/A_ACL_2022_DictBERT/glue_datasets
export outfolder=output

export TASK_NAME=cola
CUDA_VISIBLE_DEVICES=$device_num /afs/crc.nd.edu/user/w/wyu1/.conda/envs/dpt/bin/python -u finetune.py \
  --model_name_or_path $model_name \
  --task_name $TASK_NAME \
  --train_file ${input_dir}/${TASK_NAME}/train.prc.json \
  --validation_file ${input_dir}/${TASK_NAME}/validation.prc.json \
  --test_file ${input_dir}/${TASK_NAME}/test.prc.json \
  --dict_file ${input_dir}/${TASK_NAME}/vocab.90.json \
  --do_train \
  --do_eval \
  --do_predict \
  --save_steps 100000 \
  --logging_steps 100000 \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --remove_unused_columns False \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --fp16 \
  --fp16_opt_level O2 \
  --output_dir $outfolder/$TASK_NAME/

# export TASK_NAME=sst2
# CUDA_VISIBLE_DEVICES=$device_num /afs/crc.nd.edu/user/w/wyu1/anaconda3/envs/dpt/bin/python -u ft_glue.py \
#   --model_name_or_path $model_name \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --save_steps 100000 \
#   --logging_steps 100000 \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir $outfolder/$TASK_NAME/

# export TASK_NAME=mrpc
# CUDA_VISIBLE_DEVICES=$device_num /afs/crc.nd.edu/user/w/wyu1/anaconda3/envs/dpt/bin/python -u ft_glue.py \
#   --model_name_or_path $model_name \
#   --task_name $TASK_NAME \
#   --dict_file entry-${model_name}-glue.json \
#   --do_train \
#   --do_eval \
#   --save_steps 100000 \
#   --logging_steps 100000 \
#   --max_seq_length 256 \
#   --per_device_train_batch_size 32 \
#   --remove_unused_columns False \
#   --learning_rate 2e-5 \
#   --num_train_epochs 5 \
#   --fp16 \
#   --fp16_opt_level O2 \
#   --output_dir $outfolder/$TASK_NAME/

# export TASK_NAME=stsb
# CUDA_VISIBLE_DEVICES=$device_num /afs/crc.nd.edu/user/w/wyu1/anaconda3/envs/dpt/bin/python -u ft_glue.py \
#   --model_name_or_path $model_name \
#   --task_name $TASK_NAME \
#   --dict_file entry-${model_name}-glue.json \
#   --do_train \
#   --do_eval \
#   --save_steps 100000 \
#   --logging_steps 100000 \
#   --max_seq_length 256 \
#   --per_device_train_batch_size 32 \
#   --remove_unused_columns False \
#   --learning_rate 2e-5 \
#   --num_train_epochs 5 \
#   --fp16 \
#   --fp16_opt_level O2 \
#   --output_dir $outfolder/$TASK_NAME/

# export TASK_NAME=qqp
# CUDA_VISIBLE_DEVICES=$device_num /afs/crc.nd.edu/user/w/wyu1/anaconda3/envs/dpt/bin/python -u ft_glue.py \
#   --model_name_or_path $model_name \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --save_steps 100000 \
#   --logging_steps 100000 \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir $outfolder/$TASK_NAME/

# export TASK_NAME=mnli
# CUDA_VISIBLE_DEVICES=$device_num /afs/crc.nd.edu/user/w/wyu1/anaconda3/envs/dpt/bin/python -u ft_glue.py \
#   --model_name_or_path $model_name \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --save_steps 100000 \
#   --logging_steps 100000 \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir $outfolder/$TASK_NAME/

# export TASK_NAME=qnli
# CUDA_VISIBLE_DEVICES=$device_num /afs/crc.nd.edu/user/w/wyu1/anaconda3/envs/dpt/bin/python -u ft_glue.py \
#   --model_name_or_path $model_name \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --save_steps 100000 \
#   --logging_steps 100000 \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir $outfolder/$TASK_NAME/

# export TASK_NAME=rte
# CUDA_VISIBLE_DEVICES=$device_num /afs/crc.nd.edu/user/w/wyu1/anaconda3/envs/dpt/bin/python -u ft_glue.py \
#   --model_name_or_path $model_name \
#   --task_name $TASK_NAME \
#   --dict_file entry-${model_name}-glue.json \
#   --do_train \
#   --do_eval \
#   --save_steps 100000 \
#   --logging_steps 100000 \
#   --max_seq_length 256 \
#   --per_device_train_batch_size 32 \
#   --remove_unused_columns False \
#   --learning_rate 2e-5 \
#   --num_train_epochs 5 \
#   --fp16 \
#   --fp16_opt_level O2 \
#   --output_dir $outfolder/$TASK_NAME/


# export TASK_NAME=wnli
# CUDA_VISIBLE_DEVICES=$device_num /afs/crc.nd.edu/user/w/wyu1/anaconda3/envs/dpt/bin/python -u ft_glue.py \
#   --model_name_or_path $model_name \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --save_steps 100000 \
#   --logging_steps 100000 \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 5 \
#   --output_dir $outfolder/$TASK_NAME/


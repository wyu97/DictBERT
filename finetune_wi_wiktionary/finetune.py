import os, sys
import json
import random
import logging
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    set_seed,
)

from transformers.trainer_utils import is_main_process
from transformers.utils import check_min_version

from ftTrainer import ftTrainer
from ftCollator import (
    default_data_collator,
    DataCollatorWithPadding,
)

check_min_version("4.7.0.dev0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_ratios: Optional[float] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the ratio of training examples to this "
            "value if set."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."})
    dict_file: Optional[str] = field(
        default=None, metadata={"help": "A json file containing the dictionary entry with 'word, input_ids, ...'"})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            # if self.task_name not in task_to_keys.keys():
            #     self.task_name = None
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

def tokenize_dataset(datasets, data_args, model, tokenizer, is_regression, num_labels, label_list):

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name in task_to_keys.keys()
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name not in task_to_keys.keys() and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Preprocessing the datasets
    if data_args.task_name in task_to_keys.keys():
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            sentence1_key, sentence2_key = 'text', None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    use_dict = True if data_args.dict_file else False
    if use_dict:
        word2dict = defaultdict(dict)
        with open(data_args.dict_file, 'r') as f:
            for line in f.readlines():
                line = json.loads(line)
                text = line['word'] + ' is ' + line['text'] + ' [SEP]'
                line['input_ids'] = tokenizer(text, add_special_tokens=False)['input_ids']
                word2dict[line['word']] = line

    def post_process_with_dict(features):
        
        batch_input_ids, batch_token_type_ids = [], []
        batch_rows, batch_columns, batch_segments = [], [], []
        
        # for input_ids in features['input_ids']:
        for input_ids, token_type_ids in zip(features['input_ids'], features['token_type_ids']):
            # token_type_ids = [0] * len(input_ids)
            assert len(input_ids) <= data_args.max_seq_length
            if len(input_ids) == data_args.max_seq_length:
                batch_input_ids.append(input_ids)
                batch_token_type_ids.append(token_type_ids)
                batch_rows.append([])
                batch_columns.append([])
                batch_segments.append([1] * len(input_ids))
                continue

            input_tokens = [tokenizer._convert_id_to_token(id) for id in input_ids]

            word_indexes = []
            input_words = ' '.join(input_tokens).replace(' ##', '').split()
            for (i, token) in enumerate(input_tokens):
                if token.startswith("##"):
                    word_indexes[-1].append(i)
                else:
                    word_indexes.append([i])
                
            assert len(input_words) == len(word_indexes)

            rows, columns = [], []
            segments = [1] * len(token_type_ids)
            current_segment = 1

            word2seg = defaultdict(int)
            for index, word in zip(word_indexes, input_words):
                wordIndict = word2dict.get(word)
                if wordIndict:
                    if word2seg.get(word):
                        rows += index # already a list of index
                        columns += [word2seg[word]] * len(index)

                    else: # not in ...
                        def_ids = wordIndict['input_ids']
                        if len(input_ids) + len(def_ids) >= data_args.max_seq_length: 
                            print(max_seq_length, 'exceed!')
                            break # if exceed the max_length
                        
                        rows += index # already a list of index
                        input_ids += def_ids

                        current_segment += 1
                        columns += [current_segment] * len(index)
                        segments += [current_segment] * len(def_ids)
                        token_type_ids += [1] * len(def_ids)

                        word2seg[word] = current_segment

            assert len(input_ids) == len(token_type_ids) == len(segments)

            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_rows.append(rows)
            batch_columns.append(columns)
            batch_segments.append(segments)
            
        return {'input_ids': batch_input_ids,
                'rows': batch_rows,
                'columns': batch_columns,
                'segments': batch_segments,
                'token_type_ids': batch_token_type_ids,
            }
    
    def preprocess_function(examples):

        # Tokenize the texts
        args = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))
        tokenized_text = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        
        if use_dict:
            tokenized_text = post_process_with_dict(tokenized_text)

        if label_to_id is not None and "label" in examples:
            tokenized_text["labels"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        
        return tokenized_text

    datasets = datasets.map(preprocess_function, 
                            batched=True,
                            load_from_cache_file=not data_args.overwrite_cache
                            )
    return datasets


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.task_name in task_to_keys.keys() and not data_args.train_file:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    else:

        if data_args.task_name == 'mnli':
            validation_matched_file = data_args.validation_file.replace('validation', 'validation_matched')
            validation_mismatched_file = data_args.validation_file.replace('validation', 'validation_mismatched')
            data_files = {"train": data_args.train_file, 
                          "validation_matched": validation_matched_file, 
                          "validation_mismatched": validation_mismatched_file, 
                        }
        else: 
            # Loading a dataset from your local files.
            # CSV/JSON training and evaluation files are needed.
            data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                if data_args.task_name == 'mnli':
                    data_files['test_matched'] = data_args.test_file.replace('test', 'test_matched')
                    data_files['test_mismatched'] = data_args.test_file.replace('test', 'test_mismatched')
                else:
                    data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    label_list = None
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"] or data_args.task_name == "stsb"
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
        
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = BertConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = BertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    datasets = tokenize_dataset(datasets, data_args, model, tokenizer, 
                            is_regression, num_labels, label_list)
    logger.warning(datasets)
    
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_ratios is not None:
            data_args.max_train_samples = int(len(train_dataset) * data_args.max_train_ratios)
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name in task_to_keys.keys():
        metric = load_metric("glue", data_args.task_name)
    else:
        metric_acc = load_metric("accuracy")
        metric_f1 = load_metric("f1")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name in task_to_keys.keys():
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            result = metric_acc.compute(predictions=preds, references=p.label_ids)
            result['mi_f1'] = metric_f1.compute(predictions=preds, references=p.label_ids, average="micro")['f1']
            result['ma_f1'] = metric_f1.compute(predictions=preds, references=p.label_ids, average="macro")['f1']
            return result

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    training_args.task_name = data_args.task_name
    # Initialize our Trainer
    trainer = ftTrainer(
        model=model,
        args=training_args,
        dataset=datasets,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        predict_dataset=predict_dataset if training_args.do_predict else None,
        label_list=label_list if training_args.do_predict else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        
        trainer.train()
        trainer.save_state()

    if training_args.do_predict:
        
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.tsv")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()

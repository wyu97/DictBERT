import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Union

import torch
import torch.nn.functional as F
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

InputDataClass = NewType("InputDataClass", Any)
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class DataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):

        assert isinstance(features[0], (dict, BatchEncoding))

        input_ids = [f['input_ids'] for f in features]
        token_type_ids = [f['token_type_ids'] for f in features]
        segments = [f['segments'] for f in features]

        batch_input_ids = _collate_batch(input_ids, self.tokenizer)
        batch_token_type_ids = _collate_batch(token_type_ids, self.tokenizer, padding_token_id=0)
        batch_segments = _collate_batch(segments, self.tokenizer, padding_token_id=0)
        
        if features[0].get('attention_mask'):
            attention_mask = [f["attention_mask"] for f in features]
            batch_attention_mask = _collate_batch(attention_mask, self.tokenizer, padding_token_id=0)
            raise TypeError('Should not be used!')

        else: # No attention mask in training, since create it as follows
            segments = [f['segments'] for f in features]
            batch_segments = _collate_batch(segments, self.tokenizer, padding_token_id=0)
            
            nums, rows, columns = [], [], []
            for idx, f in enumerate(features):
                rows += f['rows']
                columns += f['columns']
                nums += len(f['rows']) * [idx]

            batch_one_hot = F.one_hot(batch_segments)
            batch_one_hot_T = batch_one_hot.clone().transpose(1, 2)

            batch_one_hot[nums, rows, columns] = 1
            batch_attention_mask = batch_one_hot @ batch_one_hot_T

        if features[0].get('labels') is not None:
            batch_labels = torch.tensor([f['labels'] for f in features])
        elif features[0].get('label_ids') is not None:
            batch_labels = torch.tensor([f['label_ids'] for f in features])
        elif features[0].get('label') is not None:
            batch_labels = torch.tensor([f['label'] for f in features])
        else:
            return {"input_ids": batch_input_ids, 
                "attention_mask": batch_attention_mask,
                "token_type_ids": batch_token_type_ids,
            }

        return {"input_ids": batch_input_ids, 
                "labels": batch_labels, 
                "attention_mask": batch_attention_mask,
                "token_type_ids": batch_token_type_ids,
            }


def _collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None, padding_token_id: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    
    if padding_token_id is not None:
        result = examples[0].new_full([len(examples), max_length], padding_token_id)
    else:
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)

    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


def tolist(x: Union[List[Any], torch.Tensor]):
    return x.tolist() if isinstance(x, torch.Tensor) else x


@dataclass
class DataCollatorForSeq2Seq:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


def default_data_collator(features: List[InputDataClass]):
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch

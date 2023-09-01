# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=4096):
        self.ann = [json.loads(line) for line in open(dataset_config.data_path).readlines()]
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:2]

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        ann = self.ann[index]

        in_ = None
        if "skip_prompt_formatting" in ann:
            in_ = ann["instruction"]
        else:
            if "system" in ann:
                in_ = ann["system"].strip() + "\n"
            else:
                in_ = "A chat."
                if random.random() <= 0.3:
                    in_ += " "
                else:
                    in_ += "\n"
            in_ += "USER: " + ann["instruction"].strip()
            if in_.endswith("PLAINFORMAT"):
                in_ = re.sub(r"[\n\s]+PLAINFORMAT$", "", in_, re.DOTALL)
                if random.random() <= 0.5:
                    in_ += "\nPLAINFORMAT"
                else:
                    in_ += " PLAINFORMAT"
            if random.random() <= 0.3:
                in_ += " "
            else:
                in_ += "\n"
            in_ += "ASSISTANT: "
        out_ = ann["response"].strip() + "\n"
        example = in_ + out_
        prompt = torch.tensor(
            self.tokenizer.encode(in_), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }

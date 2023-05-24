# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# we utilize the GLUE tasks listed in the below code
# https://github.com/huggingface/transformers/blob/7378726df60b9cf399aacfe372fea629c1c4c7d3/examples/pytorch/text-classification/run_glue.py

# This code follows the downstream interface of S3PRL
# https://github.com/s3prl/s3prl/tree/main/s3prl/downstream

import os

import numpy as np
import pandas as pd
from espnet2.bin.tts_inference import Text2Speech
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer

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


class GLUEDataset(Dataset):
    def __init__(self, split, glue_task, glue_root, upstream_ckpt, **kwargs):
        super(GLUEDataset, self).__init__()

        self.glue_task = glue_task
        self.glue_root = glue_root
        self.split_sets = kwargs[split]
        self.glue_dir = os.path.join(glue_root, glue_task)
        self.upstream_ckpt = upstream_ckpt

        assert os.path.isdir(
            self.glue_dir
        ), "Please first run `python downstream/glue_asr/data_prep.py -h` to get TTS version text file."

        table_list = []
        for item in self.split_sets:
            file_path = os.path.join(self.glue_dir, item, "data.csv")
            assert os.path.isfile(file_path), f"{file_path} is not found."
            table_list.append(pd.read_csv(file_path))

        self.sentence1_key, self.sentence2_key = task_to_keys[self.glue_task]
        self.df_dataset = pd.concat(table_list)
        assert len(self.df_dataset) != 0, f"0 data found for {split}"
        self.num_class = len(set(list(self.df_dataset["label"])))

        # tokenizer
        self.use_phoneme = kwargs.get("use_phoneme", False)
        if self.use_phoneme:
            print("Use phoneme tokenizer")
            # ckpt is the ESPnet TTS model name such as "kan-bayashi/ljspeech_vits"
            text2speech = Text2Speech.from_pretrained(self.upstream_ckpt, device="cuda")
            self.proc_fn = text2speech.preprocess_fn
            self.text_name = self.proc_fn.text_name
        else:
            use_fast_tokenizer = kwargs.get("use_fast_tokenizer", True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.upstream_ckpt,
                cache_dir="data",
                use_fast=use_fast_tokenizer,
            )
            self.max_seq_length = self.tokenizer.model_max_length
            # whether to distinguish the first and second sentence
            self.use_segment_emb = kwargs.get("use_segment_emb", True)
            # whether to use only text token embedding (if True, not use [CLS], [SEP]...)
            self.use_only_text_token = kwargs.get("use_only_text_token", False)

    def _x_name(self, index):
        return self.glue_task + "_" + str(index)

    def _load_text(self, index):
        texts = (
            (str(self.df_dataset[self.sentence1_key][index]),)
            if self.sentence2_key is None
            else (
                str(self.df_dataset[self.sentence1_key][index]),
                str(self.df_dataset[self.sentence2_key][index]),
            )
        )
        return texts

    def _load_bpe_token(self, args):
        if self.use_only_text_token:
            token_id_seq = self.tokenizer.tokenize(
                *args, max_length=self.max_seq_length, truncation=True
            )
            token_id_seq = np.array(
                self.tokenizer.convert_tokens_to_ids(token_id_seq)
            ).reshape(1, 1, -1)
        else:
            if self.use_segment_emb:
                token_id_seq = self.tokenizer(
                    *args,
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="np",
                )
                # not use attention_mask in order to pad later
                # keys: ['input_ids', 'token_type_ids', 'attention_mask']
                if set(token_id_seq.keys()) != set(
                    ["input_ids", "token_type_ids", "attention_mask"]
                ):
                    raise ValueError(
                        f"Invalid tokenize output keys: {token_id_seq.keys()}"
                    )
                token_id_seq = np.array(
                    [
                        token_id_seq[k]
                        for k in ["input_ids", "token_type_ids", "attention_mask"]
                    ]
                )  # KxBxT (K=3, B=1)
            else:
                token_id_seq = self.tokenizer(
                    *args,
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="np",
                )[
                    "input_ids"
                ]  # BxT
                token_id_seq = np.expand_dims(token_id_seq, 0)  # KxBxT (K=1, B=1)

        if token_id_seq.shape[1] != 1:
            raise ValueError(f"Invalid batch size ({token_id_seq.shape[1]})")
        return token_id_seq[:, 0, :].transpose(1, 0)  # KxT -> TxK

    def _load_phoneme_token(self, args):
        if self.sentence2_key is None:
            token_id_seq = self.proc_fn._text_process({self.text_name: args[0]})["text"]
        else:
            token_id_seq = np.concatenate(
                [
                    self.proc_fn._text_process({self.text_name: args[0]})["text"],
                    np.array([-1]),
                    self.proc_fn._text_process({self.text_name: args[1]})["text"],
                ]
            )
        # adding two value for [PAD] and [SEP] token (define [PAD] and [SEP] token as 0 and 1)
        token_id_seq += 2
        return token_id_seq  # T

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, index):
        label = self.df_dataset["label"][index]
        filename = self._x_name(index)
        texts = self._load_text(index)
        if self.use_phoneme:
            token_seq = self._load_phoneme_token(texts)
        else:
            token_seq = self._load_bpe_token(texts)
        return token_seq, label, filename

    def collate_fn(self, samples):
        return zip(*samples)

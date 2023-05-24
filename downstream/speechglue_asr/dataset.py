# Copyleft (c), Speech Lab, NTU, Taiwan
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# This code changes to load speechGLUE data based on the following code (and some code formatting).
# https://github.com/huggingface/transformers/blob/7378726df60b9cf399aacfe372fea629c1c4c7d3/examples/pytorch/text-classification/run_glue.py

# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# we utilize the GLUE tasks listed in the below code
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py

import os
import re

import pandas as pd
import torchaudio
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from .dictionary import Dictionary

SAMPLE_RATE = 16000
HALF_BATCHSIZE_TIME = 2000

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


####################
# Sequence Dataset #
####################
class SequenceDataset(Dataset):
    def __init__(
        self, split, bucket_size, dictionary, speechglue_task, speechglue_root, **kwargs
    ):
        super(SequenceDataset, self).__init__()

        self.dictionary = dictionary
        self.speechglue_task = speechglue_task
        self.speechglue_root = speechglue_root
        self.sample_rate = SAMPLE_RATE
        self.split_sets = kwargs[split]
        self.speechglue_dir = os.path.join(speechglue_root, speechglue_task)

        # Read table for bucketing
        assert os.path.isdir(
            self.speechglue_dir
        ), "Please first run `python downstream/speechglue_asr/data_prep.py -h` to get TTS file."

        # Wavs
        table_list = []
        for item in self.split_sets:
            file_path = os.path.join(self.speechglue_dir, item, "data.csv")
            assert os.path.isfile(file_path), f"{file_path} is not found."
            table_list.append(pd.read_csv(file_path))

        table_list = pd.concat(table_list)

        dataset_columns = ["file_path", "length", "label"]
        # the case of a dataset with a limited amount of samples in advance
        if set(table_list.columns) == set(dataset_columns):
            df_dataset = table_list
        else:
            sentence1_key, sentence2_key = task_to_keys[self.speechglue_task]
            file_paths = table_list["file_" + sentence1_key].tolist()
            labels = table_list[sentence1_key].tolist()
            lengths = table_list["length_" + sentence1_key].tolist()
            if sentence2_key is not None:
                file_paths.extend(table_list["file_" + sentence2_key].tolist())
                labels.extend(table_list[sentence2_key].tolist())
                lengths.extend(table_list["length_" + sentence2_key].tolist())
            df_dataset = pd.DataFrame(
                data={"file_path": file_paths, "length": lengths, "label": labels},
                columns=dataset_columns,
            )

        df_dataset = df_dataset.sort_values(by=["length"], ascending=False)

        X = df_dataset["file_path"].tolist()
        X_lens = df_dataset["length"].tolist()
        Y = self._load_transcript(df_dataset["label"].tolist())
        Y = [
            self.dictionary.encode_line(y, line_tokenizer=lambda x: x.split()).long()
            for y in Y
        ]
        assert len(X) != 0, f"0 data found for {split}"

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        self.Y = []
        batch_x, batch_len, batch_y = [], [], []

        for x, x_len, y in tqdm(
            zip(X, X_lens, Y),
            total=len(X),
            desc=f"ASR dataset {split}",
            dynamic_ncols=True,
        ):
            batch_x.append(x)
            batch_len.append(x_len)
            batch_y.append(y)

            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                    self.X.append(batch_x[: bucket_size // 2])
                    self.X.append(batch_x[bucket_size // 2 :])
                    self.Y.append(batch_y[: bucket_size // 2])
                    self.Y.append(batch_y[bucket_size // 2 :])
                else:
                    self.X.append(batch_x)
                    self.Y.append(batch_y)
                batch_x, batch_len, batch_y = [], [], []

        # Gather the last batch
        if len(batch_x) > 1:
            self.X.append(batch_x)
            self.Y.append(batch_y)

    def _parse_x_name(self, x):
        return "-".join(x.split("/")[-4:]).split(".")[0]

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        assert (
            sr == self.sample_rate
        ), f"Sample rate mismatch: real {sr}, config {self.sample_rate}"
        return wav.view(-1)

    def _load_transcript(self, x_list):
        def process_trans(transcript):
            transcript = re.sub("[.,?!]", "", transcript).replace(" ", "|")
            # word to char
            return " ".join(list(transcript)) + " |"

        return [process_trans(x) for x in x_list]

    def _build_dictionary(
        self, transcripts, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        d = Dictionary()
        transcript_list = list(transcripts.values())
        Dictionary.add_transcripts_to_dictionary(transcript_list, d, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav_batch = [self._load_wav(x_file).numpy() for x_file in self.X[index]]
        label_batch = [y.numpy() for y in self.Y[index]]
        filename_batch = [self._parse_x_name(x_file) for x_file in self.X[index]]
        return (
            wav_batch,
            label_batch,
            filename_batch,
        )  # bucketing, return ((wavs, labels))

    def collate_fn(self, items):
        assert len(items) == 1
        return (
            items[0][0],
            items[0][1],
            items[0][2],
        )  # hack bucketing, return (wavs, labels, filenames)

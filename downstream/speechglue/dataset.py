# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# we utilize the GLUE tasks listed in the below code
# https://github.com/huggingface/transformers/blob/7378726df60b9cf399aacfe372fea629c1c4c7d3/examples/pytorch/text-classification/run_glue.py

# This code follows the downstream interface of S3PRL
# https://github.com/s3prl/s3prl/tree/main/s3prl/downstream

import os

import pandas as pd
import torch
import torchaudio
from torch.utils.data.dataset import Dataset

SAMPLE_RATE = 16000
SEP_DURATION = int(0.05 * 16000)  # 50 ms

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


class SpeechGLUEDataset(Dataset):
    def __init__(self, split, speechglue_task, speechglue_root, **kwargs):
        super(SpeechGLUEDataset, self).__init__()

        self.speechglue_task = speechglue_task
        self.speechglue_root = speechglue_root
        self.sample_rate = SAMPLE_RATE
        self.split_sets = kwargs[split]
        self.speechglue_dir = os.path.join(speechglue_root, speechglue_task)
        # use a fixed random signal
        sep_sig_length = kwargs.get("sep_sig_length", 50)
        if sep_sig_length == 50:
            self.sep_sig_path = os.path.join(
                "downstream", "speechglue", "white_noise.wav"
            )
        else:
            print(f"Use {sep_sig_length}ms SEP signal")
            self.sep_sig_path = os.path.join(
                "dump", f"white_noise_{sep_sig_length}ms.wav"
            )
        self.late_concat = kwargs.get("late_concat", False)

        assert os.path.isdir(
            self.speechglue_dir
        ), "Please first run `python downstream/speechglue_asr/data_prep.py -h` to get TTS file."

        table_list = []
        for item in self.split_sets:
            file_path = os.path.join(self.speechglue_dir, item, "data.csv")
            assert os.path.isfile(file_path), f"{file_path} is not found."
            table_list.append(pd.read_csv(file_path))

        self.sentence1_key, self.sentence2_key = task_to_keys[self.speechglue_task]
        self.df_dataset = pd.concat(table_list)
        assert len(self.df_dataset) != 0, f"0 data found for {split}"
        self.num_class = len(set(list(self.df_dataset["label"])))

        if not self.late_concat:
            self.sep_sig, sr = torchaudio.load(self.sep_sig_path)

    def _x_name(self, index):
        return self.speechglue_task + "_" + str(index)

    def _load_wav(self, index):
        wav1, sr = torchaudio.load(self.df_dataset["file_" + self.sentence1_key][index])
        assert (
            sr == self.sample_rate
        ), f"Sample rate mismatch: real {sr}, config {self.sample_rate}"
        if self.sentence2_key is None:
            return wav1.view(-1)
        else:
            wav2, sr = torchaudio.load(
                self.df_dataset["file_" + self.sentence2_key][index]
            )
            assert (
                sr == self.sample_rate
            ), f"Sample rate mismatch: real {sr}, config {self.sample_rate}"
            if not self.late_concat:
                sep_sig = self.sep_sig.to(device=wav1.device, dtype=wav1.dtype)
                return torch.cat((wav1.view(-1), sep_sig.view(-1), wav2.view(-1)))
            else:
                return wav1.view(-1), wav2.view(-1)

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, index):
        label = self.df_dataset["label"][index]
        filename = self._x_name(index)
        if not self.late_concat:
            wav = self._load_wav(index).numpy()
            return wav, label, filename
        else:
            wav1, wav2 = self._load_wav(index)
            return wav1.numpy(), wav2.numpy(), label, filename

    def collate_fn(self, samples):
        if not self.late_concat:
            return zip(*samples)
        else:
            wavs1, wavs2, labels, filenames = zip(*samples)
            all_wavs = wavs1 + wavs2
            return all_wavs, labels, filenames

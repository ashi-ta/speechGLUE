import logging

import torch
import torch.nn as nn
import yaml
from espnet2.bin.tts_inference import Text2Speech
from torch.nn.utils.rnn import pad_sequence

SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, model_config=None, **kwds):
        super().__init__()
        # ckpt is the ESPnet TTS model name such as "kan-bayashi/ljspeech_vits"
        text2speech = Text2Speech.from_pretrained(ckpt, device="cuda")
        # adding two value for [PAD] and [SEP] token (define [PAD] and [SEP] token as 0 and 1)
        self.vocab_size = (
            len(text2speech.preprocess_fn.token_id_converter.token_list) + 2
        )
        print(f"Phoneme vocabulary size is {self.vocab_size}")

        if model_config is not None:
            print(
                "[UpstreamExpert] - Using upstream expert config file from:",
                model_config,
            )
            with open(model_config, "r") as file:
                options = yaml.load(file, Loader=yaml.FullLoader)
        else:
            print("[UpstreamExpert] - Using the default upstream expert config")
            options = {
                "embedding_size": 256,
            }
        self.model = nn.Embedding(
            self.vocab_size, options["embedding_size"], padding_idx=0
        )
        print(f"Embedding size is {options['embedding_size']}")

    def get_downsample_rates(self, key: str = None) -> int:
        return 1

    def forward(self, tokens):
        # tokens: List of FloatTensor(T)
        # when Featurizer instantiation, tokens is List of FloatTensor(T)
        # https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/interfaces.py
        if tokens[0].dim() == 1 and tokens[0].shape[0] == SAMPLE_RATE:
            print("Featurizer instantiation related forward")
            tokens[0] = torch.randint(
                0, self.vocab_size, (20, 1), device=tokens[0].device
            )
        padded_token = pad_sequence(tokens, batch_first=True, padding_value=0).to(
            dtype=torch.int64
        )  # BxT
        output_values = self.model(padded_token)  # BxTxF
        return {"hidden_states": output_values}

import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

SAMPLE_RATE = 16000
HF_INPUT_KEYS = ["input_ids", "token_type_ids", "attention_mask"]

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        self.model = AutoModel.from_pretrained(ckpt)
        tokenizer = AutoTokenizer.from_pretrained(
            ckpt,
            cache_dir="data",
        )
        self.pad_values = [tokenizer.pad_token_id, tokenizer.pad_token_type_id, 0]
        self.vocab_size = tokenizer.vocab_size

    def pad_token(self, tokens):
        # https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py
        device = tokens[0].device
        key_size = tokens[0].shape[1]
        output_dict = {}
        for key_id in range(key_size):
            key_name = HF_INPUT_KEYS[key_id]
            padded_token = pad_sequence(
                [token[:, key_id] for token in tokens],
                batch_first=True,
                padding_value=self.pad_values[key_id],
            )
            output_dict[key_name] = padded_token.to(dtype=torch.int64, device=device)
        return output_dict

    def get_downsample_rates(self, key: str = None) -> int:
        return 1

    def forward(self, tokens):
        # tokens: List of FloatTensor(TxK)
        # when Featurizer instantiation, tokens is List of FloatTensor(T)
        # https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/interfaces.py
        if tokens[0].dim() == 1 and tokens[0].shape[0] == SAMPLE_RATE:
            print("Featurizer instantiation related forward")
            tokens[0] = torch.randint(
                0, self.vocab_size, (20, 1), device=tokens[0].device
            )
        input_dict = self.pad_token(tokens)
        output_values = self.model(
            **input_dict, output_hidden_states=True
        )  # Tuple of BxTxF
        return {"hidden_states": output_values.hidden_states}

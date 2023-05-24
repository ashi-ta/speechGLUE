import torch.nn as nn

# import the various pooling layers from:
# https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/model.py
from ..model import *


class SequenceClassifierWithDropout(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        pooling="MeanPooling",
        activation="ReLU",
        dropout=0.1,
        pooling_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pooling_dim = pooling_dim
        if pooling_dim is None:
            pooling_dim = input_dim
        else:
            self.pooling_linear = nn.Linear(input_dim, pooling_dim)
            self.pooling_activation = nn.Tanh()
        self.pooler = eval(pooling)(input_dim=input_dim, activation=activation)
        self.classifier = nn.Linear(pooling_dim, output_dim)

    def forward(self, hidden_state, features_len=None):

        pooled_tensor, features_len = self.pooler(hidden_state, features_len)
        if self.pooling_dim is not None:
            pooled_tensor = self.pooling_linear(pooled_tensor)
            pooled_tensor = self.pooling_activation(pooled_tensor)
        pooled_tensor = self.dropout(pooled_tensor)
        logit = self.classifier(pooled_tensor)
        return logit, None

import torch.nn as nn


class SequenceClassifierWithCLSPooling(nn.Module):
    def __init__(self, input_dim, output_dim, pooling_dim=None, dropout=0.1, **kwargs):
        super(SequenceClassifierWithCLSPooling, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pooling_dim = pooling_dim
        if pooling_dim is None:
            pooling_dim = input_dim
        else:
            self.pooling_linear = nn.Linear(input_dim, pooling_dim)
            self.pooling_activation = nn.Tanh()
        self.classifier = nn.Linear(pooling_dim, output_dim)

    def forward(self, features, features_len=None):
        # features: BxTxF
        # The first token output corresponding to [CLS] token for the BERT model
        cls_tensor = features[:, 0, :]
        if self.pooling_dim is not None:
            cls_tensor = self.pooling_linear(cls_tensor)
            cls_tensor = self.pooling_activation(cls_tensor)
        cls_tensor = self.dropout(cls_tensor)
        logits = self.classifier(cls_tensor)
        return logits, None

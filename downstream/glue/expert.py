# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# we utilize the GLUE tasks listed in the below code
# https://github.com/huggingface/transformers/blob/7378726df60b9cf399aacfe372fea629c1c4c7d3/examples/pytorch/text-classification/run_glue.py

# This code follows the downstream interface of S3PRL
# https://github.com/s3prl/s3prl/tree/main/s3prl/downstream

from pathlib import Path

import evaluate
import torch
import torch.nn as nn
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler

from ..model import *
from .dataset import GLUEDataset
from .model import *

task_to_metrics = {
    "cola": ("matthews_correlation", None),
    "mnli": ("accuracy", None),
    "mrpc": ("accuracy", "f1"),
    "qnli": ("accuracy", None),
    "qqp": ("accuracy", "f1"),
    "rte": ("accuracy", None),
    "sst2": ("accuracy", None),
    "stsb": ("pearson", "spearmanr"),
    "wnli": ("accuracy", None),
}


class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = expdir
        self.upstream_ckpt = kwargs["upstream_ckpt"]

        # define a task
        self.glue_task = self.datarc["glue_task"]
        self.is_regression = self.glue_task == "stsb"
        if not self.is_regression:
            self.objective = nn.CrossEntropyLoss()
            self.num_class = GLUEDataset(
                "train", upstream_ckpt=self.upstream_ckpt, **self.datarc
            ).num_class
        else:
            self.objective = nn.MSELoss()
            self.num_class = 1
            print(f"{self.glue_task} will be executed as a regression task")

        model_cls = eval(self.modelrc["select"])
        model_conf = self.modelrc.get(self.modelrc["select"], {})
        projector_dim = self.modelrc.get("projector_dim", None)
        if projector_dim is not None:
            self.projector = nn.Linear(upstream_dim, self.modelrc["projector_dim"])
            model_input_dim = projector_dim
        else:
            self.projector = None
            model_input_dim = upstream_dim
        self.model = model_cls(
            input_dim=model_input_dim,
            output_dim=self.num_class,
            **model_conf,
        )

        self.normalize = self.modelrc.get("tanh_normalization", False)
        if self.normalize:
            print("Use Tanh normalization")
            self.norm_act_fn = nn.Tanh()

        self.dropout = self.modelrc.get("dropout", None)
        if self.dropout is not None:
            print("Use dropout after projection")
            self.dropout = nn.Dropout(self.dropout)

        self.metric = evaluate.load("glue", self.glue_task)
        self.metric_keys1, self.metric_keys2 = task_to_metrics[self.glue_task]
        self.expdir = expdir
        self.register_buffer("best_metric1_score", torch.zeros(1))
        self.register_buffer("best_metric2_score", torch.zeros(1))

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=self.datarc["train_batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc["eval_batch_size"],
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def get_dataloader(self, split):
        if not hasattr(self, f"{split}_dataset"):
            setattr(
                self,
                f"{split}_dataset",
                GLUEDataset(split, upstream_ckpt=self.upstream_ckpt, **self.datarc),
            )

        if split == "train":
            return self._get_train_dataloader(self.train_dataset)
        else:
            return self._get_eval_dataloader(getattr(self, f"{split}_dataset"))

    # Interface
    def forward(self, mode, features, labels, filenames, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(
            device=device
        )

        features = pad_sequence(features, batch_first=True)
        if self.projector is not None:
            features = self.projector(features)
        if self.normalize:
            features = self.norm_act_fn(features)
        if self.dropout is not None:
            features = self.dropout(features)
        predicted, _ = self.model(features, features_len)

        if not self.is_regression:
            labels = torch.LongTensor(labels).to(features.device).view(-1)
            predicted = predicted.view(-1, self.num_class)
            predicted_id_value = torch.argmax(predicted, dim=-1)
        else:
            labels = torch.FloatTensor(labels).to(features.device).squeeze()
            predicted = predicted.squeeze()
            predicted_id_value = predicted

        loss = self.objective(predicted, labels)

        records["loss"].append(loss.item())
        records["filename"] += filenames
        records["predict"] += predicted_id_value.cpu().flatten().tolist()
        records["truth"] += labels.cpu().flatten().tolist()

        return loss

    def dump_prediction(self, outpath, filename, pred, label, step=0):
        with open(outpath, "w") as file:
            line = [f"{step},{f},{p},{l}\n" for f, p, l in zip(filename, pred, label)]
            file.writelines(line)

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        dev_update1 = False
        dev_update2 = False
        save_names = []

        # loss related
        values = records["loss"]
        loss_average = torch.FloatTensor(values).mean().item()
        logger.add_scalar(
            f"glue-{self.glue_task}/{mode}-loss",
            loss_average,
            global_step=global_step,
        )

        # score related
        results = self.metric.compute(
            predictions=records["predict"], references=records["truth"]
        )
        print(f"{mode}: {results}")
        for k in [self.metric_keys1, self.metric_keys2]:
            if k is None:
                continue
            result = results[k]
            logger.add_scalar(
                f"glue-{self.glue_task}/{mode}-{k}",
                result,
                global_step=global_step,
            )
            with open(Path(self.expdir) / "train.csv", "a") as f:
                f.write(f"{mode},{global_step},{loss_average},{k},{result}\n")
                if mode == "dev":
                    if result > self.best_metric1_score and k == self.metric_keys1:
                        dev_update1 = True
                        self.best_metric1_score = torch.ones(1) * result
                        f.write(
                            f"{mode},{global_step},{loss_average},{k}_update,{result}\n"
                        )
                        save_names.append(f"{mode}-{k}-best.ckpt")

                    if result > self.best_metric2_score and k == self.metric_keys2:
                        dev_update2 = True
                        self.best_metric2_score = torch.ones(1) * result
                        f.write(
                            f"{mode},{global_step},{loss_average},{k}_update,{result}\n"
                        )
                        save_names.append(f"{mode}-{k}-best.ckpt")

        if mode == "test":
            self.dump_prediction(
                Path(self.expdir) / f"dump_{mode}.csv",
                records["filename"],
                records["predict"],
                records["truth"],
            )
        elif mode == "dev" and dev_update1:
            self.dump_prediction(
                Path(self.expdir) / f"dump_{mode}_{self.metric_keys1}_best.csv",
                records["filename"],
                records["predict"],
                records["truth"],
                global_step,
            )
        elif mode == "dev" and dev_update2:
            self.dump_prediction(
                Path(self.expdir) / f"dump_{mode}_{self.metric_keys2}_best.csv",
                records["filename"],
                records["predict"],
                records["truth"],
                global_step,
            )

        return save_names

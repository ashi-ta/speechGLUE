# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# we utilize the GLUE tasks listed in the below code
# https://github.com/huggingface/transformers/blob/7378726df60b9cf399aacfe372fea629c1c4c7d3/examples/pytorch/text-classification/run_glue.py

import argparse
import os
import random

import pandas as pd

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


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Data preparation for SpeechGLUE",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="dump",
        help="Path to storing the SpeechGLUE dataset",
    )
    parser.add_argument(
        "--glue-task",
        type=str,
        default="all",
        choices=["all"] + list(task_to_keys.keys()),
        help="Name of the GLUE task",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Split of dataset",
    )
    parser.add_argument(
        "--max-hours",
        type=int,
        default=None,
        help="Upper limit of time in hours",
    )
    parser.add_argument(
        "--no-use-predefined-sampling",
        action="store_true",
        help="No predefined sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed value",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    max_samples = int(args.max_hours * 60 * 60 * 16000)

    if args.glue_task == "all":
        task_names = task_to_keys.keys()
    else:
        task_names = [args.glue_task]

    for task_name in task_names:
        # set a data split
        if args.split == "train":
            data_splits = ["train"]
        elif args.split == "validation":
            data_splits = (
                ["validation_matched", "validation_mismatched"]
                if task_name == "mnli"
                else ["validation"]
            )
        elif args.split == "test":
            data_splits = (
                ["test_matched", "test_mismatched"] if task_name == "mnli" else ["test"]
            )
        else:
            raise ValueError(
                "args.split must be one of the ['train', 'validation', 'test']"
            )

        for data_split in data_splits:
            file_path = os.path.join(args.dump_dir, task_name, data_split, "data.csv")
            assert os.path.isfile(file_path), f"{file_path} is not found."
            table = pd.read_csv(file_path)

            sentence1_key, sentence2_key = task_to_keys[task_name]
            file_paths = table["file_" + sentence1_key].tolist()
            labels = table[sentence1_key].tolist()
            lengths = table["length_" + sentence1_key].tolist()
            if sentence2_key is not None:
                file_paths.extend(table["file_" + sentence2_key].tolist())
                labels.extend(table[sentence2_key].tolist())
                lengths.extend(table["length_" + sentence2_key].tolist())

            if sum(lengths) < max_samples:
                current_hours = round(sum(lengths) / 16000 / 60 / 60, 1)
                print(
                    f"The {data_split} set of {task_name} task ({current_hours}) is already less than {args.max_hours} hours."
                )
                continue

            select_outdir = os.path.join(
                "downstream", "speechglue_asr", "selected_uttids"
            )
            select_filename = task_name + "_" + data_split + ".list"
            if args.no_use_predefined_sampling:
                uttids = list(range(len(file_paths)))
                random.shuffle(uttids)
            else:
                with open(os.path.join(select_outdir, select_filename), "r") as f:
                    uttids = [int(i) for i in f.read().splitlines()]

            num_sample = 0
            file_paths_lim = []
            labels_lim = []
            lengths_lim = []
            uttids_lim = []
            for uttid in uttids:
                num_sample += lengths[uttid]
                if num_sample < max_samples:
                    file_paths_lim.append(file_paths[uttid])
                    labels_lim.append(labels[uttid])
                    lengths_lim.append(lengths[uttid])
                    uttids_lim.append(uttid)
                else:
                    break

            df_dataset = pd.DataFrame(
                data={
                    "file_path": file_paths_lim,
                    "length": lengths_lim,
                    "label": labels_lim,
                },
                columns=["file_path", "length", "label"],
            )
            df_outdir = os.path.join(
                args.dump_dir, task_name, data_split + "-" + str(args.max_hours)
            )
            os.makedirs(df_outdir, exist_ok=True)
            df_dataset.to_csv(os.path.join(df_outdir, "data.csv"), index=False)

            if args.no_use_predefined_sampling:
                os.makedirs(select_outdir, exist_ok=True)
                with open(os.path.join(select_outdir, select_filename), "w") as f:
                    f.writelines([str(l) + os.linesep for l in uttids_lim])


if __name__ == "__main__":
    main()

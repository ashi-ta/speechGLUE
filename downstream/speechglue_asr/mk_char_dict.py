# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# we utilize the GLUE tasks listed in the below code
# https://github.com/huggingface/transformers/blob/7378726df60b9cf399aacfe372fea629c1c4c7d3/examples/pytorch/text-classification/run_glue.py

import argparse
import os
import re

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
        description="Dictionary preparation for SpeechGLUE",
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

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.glue_task == "all":
        task_names = task_to_keys.keys()
    else:
        task_names = [args.glue_task]

    for task_name in task_names:
        sentence1_key, sentence2_key = task_to_keys[task_name]
        csv_path = os.path.join(args.dump_dir, task_name, "train", "data.csv")
        # some sentences include only "null"
        # therefore, keep_default_na is added to interpret as is
        csv = pd.read_csv(csv_path, keep_default_na=False)
        sentences = list(csv[sentence1_key])
        if sentence2_key is not None:
            sentences.extend(list(csv[sentence2_key]))
        sentences = "|".join(sentences)
        sentences = re.sub("[.,?!]", "", sentences).replace(" ", "|") + "|"
        char_counts = {c: sentences.count(c) for c in set(sentences)}
        outdic = os.path.join(args.dump_dir, task_name, "char.dict")

        with open(outdic, "w") as f:
            for x in sorted(
                char_counts.items(), key=lambda char: char[1], reverse=True
            ):
                f.write(x[0] + " " + str(x[1]) + "\n")


if __name__ == "__main__":
    main()

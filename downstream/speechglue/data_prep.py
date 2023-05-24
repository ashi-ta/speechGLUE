# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# we utilize the GLUE tasks listed in the below code
# https://github.com/huggingface/transformers/blob/7378726df60b9cf399aacfe372fea629c1c4c7d3/examples/pytorch/text-classification/run_glue.py

import argparse
import logging
import math
import os
import re
import unicodedata
from builtins import str as unicode

import numpy as np
import soundfile
import tacotron_cleaner.cleaners
from datasets import load_dataset
from espnet2.bin.tts_inference import Text2Speech
from inflect import NumOutOfRangeError
from librosa import resample

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
    parser.add_argument("--verbose", "-V", default=1, type=int, help="Verbose option")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to storing the original GLUE dataset downloaded from huggingface.co",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="dump",
        help="Path to storing the SpeechGLUE dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Pytorch device",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers for map() of TTS",
    )
    parser.add_argument(
        "--glue-task",
        type=str,
        default="all",
        choices=["all"] + list(task_to_keys.keys()),
        help="Name of the GLUE task for synthesizing",
    )
    parser.add_argument(
        "--max-tts-sample",
        type=int,
        default=None,
        help="Number of limited examples for synthesizing (for testing purposes only)",
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        default="kan-bayashi/ljspeech_vits",
        help="Name of ESPnet TTS model (listed in https://github.com/espnet/espnet_model_zoo)",
    )
    return parser


def text_normalization(text, idx=0):
    # espnet-TTS uses the preprocessing sequence of text-cleaner & g2p
    # e.g. kan-bayashi/ljspeech_vits configure with tacotron cleaner & g2p_en_no_space
    # therefore, this code also uses same text-cleaner & text-processing in a g2p
    # https://github.com/espnet/espnet_tts_frontend/blob/master/tacotron_cleaner/cleaners.py
    # https://github.com/Kyubyong/g2p/blob/master/g2p_en/g2p.py

    def _space_normalization(text_with_space):
        # text normalization related with a space
        t_list = text_with_space.split()
        norm_list = []
        i = 0
        while i < len(t_list):
            if i < len(t_list) - 1:
                # merge two words with an apostrophe (e.g., "can' t" -> "can't")
                if t_list[i + 1][0] == "'":
                    norm_list.append(t_list[i] + t_list[i + 1])
                    i += 1
                # add space after comma (e.g., ",2000" -> ", 2000")
                elif t_list[i + 1][0] == ",":
                    if t_list[i + 1] == ",":
                        norm_list.extend([t_list[i] + ","])
                    else:
                        norm_list.extend([t_list[i] + ",", t_list[i + 1][1:].strip()])
                    i += 1
                # add space after period (e.g., ".2000" -> ". 2000")
                elif t_list[i + 1][0] == ".":
                    if t_list[i + 1] == ".":
                        norm_list.extend([t_list[i] + "."])
                    else:
                        norm_list.extend([t_list[i] + ".", t_list[i + 1][1:].strip()])
                    i += 1
                else:
                    norm_list.append(t_list[i])
            else:
                norm_list.append(t_list[i])
            i += 1
        return " ".join(norm_list)

    norm_text = _space_normalization(
        text.replace(". . .", ".").replace("...", ".").replace("$,", "$")
    )
    try:
        norm_text = tacotron_cleaner.cleaners.custom_english_cleaners(norm_text)
        # from https://github.com/Kyubyong/g2p/blob/master/g2p_en/g2p.py#L148
        # but normalize_numbers() has already been applied in the custom_english_cleaners()
        norm_text = unicode(norm_text)
        norm_text = "".join(
            char
            for char in unicodedata.normalize("NFD", norm_text)
            if unicodedata.category(char) != "Mn"
        )
        norm_text = norm_text.lower()
        norm_text = re.sub("[^ a-z'.,?!\-]", "", norm_text)
        norm_text = norm_text.replace("i.e.", "that is")
        norm_text = norm_text.replace("e.g.", "for example")
        # space-related normalization again after removing some punctuations
        norm_text = _space_normalization(norm_text)
    except (RuntimeError, NumOutOfRangeError) as e:
        # Some sentences can't be tokenized to vocode.
        # E.g., contain only symbols such as "(" and ")"
        # E.g., contain a number out of range (i.e., NumOutOfRangeError of https://github.com/jaraco/inflect)
        logging.warning(
            f"{e}\n"
            + "Invalid sentence may be inputted and this column will be deleted."
            + f" (sentence: {text}, idx: {idx})"
        )
        norm_text = None

    if norm_text == "":
        norm_text = None
        logging.warning(
            "Invalid sentence may be inputted and this column will be deleted."
            + f" (sentence: {text}, idx: {idx})"
        )
    return norm_text


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Setup logging
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check args
    if args.num_workers > 1 and args.device == "cuda":
        logging.warning("only single GPU decoding is supported")
        args.num_workers = 1

    # instantiate the text-to-speech model
    text2speech = Text2Speech.from_pretrained(args.tts_model, device=args.device)

    if args.glue_task == "all":
        task_names = task_to_keys.keys()
    else:
        task_names = [args.glue_task]

    for task_name in task_names:
        logging.info("[" + task_name + "] Start dataset preparation")
        # set a data split
        valid_column = "validation_matched" if task_name == "mnli" else "validation"
        eval_column = "test_matched" if task_name == "mnli" else "test"
        extend_column = (
            ["validation_mismatched", "test_mismatched"] if task_name == "mnli" else []
        )
        data_splits = ["train", valid_column, eval_column] + extend_column
        logging.info("[" + task_name + "] Splits of dataset = " + str(data_splits))

        sentence1_key, sentence2_key = task_to_keys[task_name]

        # TTS function applied by a map()
        def tts(examples):
            # first sentence
            dirname = str(math.floor(examples["idx"] / 10000) * 10000)
            outdir_base = os.path.abspath(
                os.path.join(args.dump_dir, task_name, data_split)
            )
            wav_name = str(examples["idx"]) + ".wav"
            out_path1 = os.path.join(outdir_base, sentence1_key, dirname, wav_name)
            text1 = text_normalization(examples[sentence1_key], examples["idx"])
            if text1 is None:
                out_path1 = None
                length1 = None
            else:
                speech = text2speech(text1)["wav"]
                speech = resample(
                    speech.cpu().numpy(),
                    orig_sr=text2speech.fs,
                    target_sr=16000,
                    res_type="kaiser_best",
                )
                length1 = speech.shape[0]
                soundfile.write(out_path1, speech, 16000, "PCM_16")

            if sentence2_key is None:
                return {
                    sentence1_key: text1,
                    "file_" + sentence1_key: out_path1,
                    "length_" + sentence1_key: length1,
                }

            # second sentence
            out_path2 = os.path.join(outdir_base, sentence2_key, dirname, wav_name)
            text2 = text_normalization(examples[sentence2_key], examples["idx"])
            if text2 is None:
                out_path1 = None  # for filtering
                out_path2 = None
                length1 = None
                length2 = None
            else:
                speech = text2speech(text2)["wav"]
                speech = resample(
                    speech.cpu().numpy(),
                    orig_sr=text2speech.fs,
                    target_sr=16000,
                    res_type="kaiser_best",
                )
                length2 = speech.shape[0]
                soundfile.write(out_path2, speech, 16000, "PCM_16")
            return {
                sentence1_key: text1,
                sentence2_key: text2,
                "file_" + sentence1_key: out_path1,
                "file_" + sentence2_key: out_path2,
                "length_" + sentence1_key: length1,
                "length_" + sentence2_key: length2,
            }

        # initialize a dataset and generate synthesized speech data
        logging.info("[" + task_name + "] Generating TTS data")
        for data_split in data_splits:
            # take a dataset from HuggingFace's GLUE
            raw_datasets = load_dataset(
                "glue",
                task_name,
                split=data_split,
                cache_dir=args.data_dir,
                use_auth_token=None,
            )
            num_utt = len(raw_datasets)
            logging.info(
                f"The number of rows of the original data of {data_split} split: {num_utt}"
            )

            # make output directories
            dirnames = np.arange(0, math.floor(num_utt / 10000) * 10000 + 1, 10000)
            for dirname in dirnames:
                os.makedirs(
                    os.path.join(
                        args.dump_dir,
                        task_name,
                        data_split,
                        sentence1_key,
                        str(dirname),
                    ),
                    exist_ok=True,
                )
                if sentence2_key is not None:
                    os.makedirs(
                        os.path.join(
                            args.dump_dir,
                            task_name,
                            data_split,
                            sentence2_key,
                            str(dirname),
                        ),
                        exist_ok=True,
                    )

            # limit the number of examples for testing
            if args.max_tts_sample is not None:
                raw_datasets = raw_datasets.select(range(args.max_tts_sample))
            # run a text-to-speech
            tts_datasets = raw_datasets.map(
                tts,
                num_proc=args.num_workers,
                desc="Running TTS on the "
                + data_split
                + " set of "
                + task_name
                + " dataset",
            )
            # filter rows that could not TTS
            tts_datasets = tts_datasets.filter(
                lambda example: example["file_" + sentence1_key] is not None
            )
            logging.info(
                f"The number of rows of the synthesized data of {data_split} split: {len(tts_datasets)}\n"
                + "-----------------------"
            )
            # save the audio files with CSV format
            tts_datasets.to_csv(
                os.path.join(args.dump_dir, task_name, data_split, "data.csv"),
                index=False,
            )
        logging.info("[" + task_name + "] Successfully finished dataset preparation")
    logging.info("All dataset preparation finished successfully")


if __name__ == "__main__":
    main()

# SpeechGLUE

SpeechGLUE is a speech version of the [General Language Understanding Evaluation (GLUE)](https://arxiv.org/abs/1804.07461) benchmark.
This extension is based on [S3PRL](https://github.com/s3prl/s3prl) developed for [SUPERB](https://arxiv.org/abs/2105.01051), which facilitates comparisons with various speech self-supervised learning (SSL) models.
Since GLUE comprises a variety of natural language understanding tasks, SpeechGLUE can elucidate the degree of linguistic ability of speech SSL models.
For the conversion of input text to speech, we adopt text-to-speech (TTS) systems, which allow for the realization of tasks that assess purely linguistic knowledge by constraining acoustic conditions such as variations in speakers, speaking styles and recording settings.

<p align="center">
<img src="https://github.com/ashi-ta/speechGLUE/blob/main/images/system.png" width="500px">
</p>

## Requirements

We have confirmed running in the following environments.

- [S3PRL](https://github.com/s3prl/s3prl) == v0.4.10
- [ESPnet](https://github.com/espnet/espnet) == v.202301
- Hugging Face related toolkit
  - [Transformers](https://github.com/huggingface/transformers) == v4.25.1
  - [Datasets](https://github.com/huggingface/datasets) == 2.8.0
  - [Evaluate](https://github.com/huggingface/evaluate) == v0.4.0

## Usage

### Extend the S3PRL toolkit

First, extend the original S3PRL toolkit by copying files from this repository:

```shell
git clone https://github.com/s3prl/s3prl -b v0.4.10
git clone https://github.com/ashi-ta/speechGLUE
cd s3prl/s3prl
cp -r ../../speechGLUE/hub.py ../../speechGLUE/upstream ../../speechGLUE/downstream ./
```

### Data preparation

Next, download the original GLUE dataset from Hugging Face and run TTS:

```shell
# e.g., generate speech data of COLA task
TASK=cola
python downstream/speechglue/data_prep.py ${TASK}
# or generate speech data for all tasks sequentially
python downstream/speechglue/data_prep.py
```

This generation process takes approximately the following time on a single GPU:

|  Task  |  Task key  |  Train (hours)  |  Validation (hours)  |  Test (hours)  |
| :---- | :---- | ----: | ----: | ----: |
|  CoLA  |  cola  |  0.3  |  < 0.1  |  < 0.1  |
|  SST2  |  sst2  |  2.6  |  < 0.1  |  0.1  |
|  MRPC  |  mrpc  |  0.5  |  < 0.1  |  0.3  |
|  QQP  |  qqp  |  31.4  |  3.5  |  34.2  |
|  STS-B  |  stsb  |  0.5  |  0.1  |  0.1  |
|  MNLI  |  mnli  |  41.2  |  1.0 (matched) / 1.0 (mismatched)  |  1.0 (matched) / 1.2 (mismatched)  |
|  QNLI  |  qnli  |  14.9  |  0.8  |  0.8  |
|  RTE  |  rte  |  0.5  |  < 0.1  |  0.6  |
|  WNLI  |  wnli  |  < 0.1  |  < 0.1  |  < 0.1  |

### Run SpeechGLUE benchmark

Since this extension is based on S3PRL, the running command and options follow S3PRL:

```shell
# e.g., evaluate the HuBERT model with base architecture on CoLA task
TASK=cola
UPSTREAM=hubert_base
OUTDIR=result/${TASK}/${UPSTREAM}

python run_downstream.py \
  -m train \
  -p ${OUTDIR} \
  -u ${UPSTREAM} \
  -d speechglue \
  -c downstream/speechglue/config_${TASK}.yaml
# to evaluate on other tasks, specify the task key in the above table
```

You can find other upstream SSL models (and the FBANK input baseline) from [S3PRL](https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/README.md).  
After the training, the output CSV file (located at ${OUTDIR)/train.csv) contains the results for the development set, and the best performance can be viewed using the following commands:

```shell
# for the tasks evaluated with a single metric (i.e., cola, sst2, mnli-m/-mm, qnli, rte, and wnli)
grep update ${OUTDIR}/train.csv | tail -n 1 | cut -d ',' -f 5
# for the tasks evaluated with accuracy and F1 (i.e., mrpc and qqp)
# first and second values represent accuracy and F1 scores
grep -A1 accuracy_update ${OUTDIR}/train.csv | tail -n 2 | cut -d ',' -f 5
# for the stsb task
# first and second values represent Pearson and Spearman correlation coefficients
grep -A1 pearson_update ${OUTDIR}/train.csv | tail -n 2 | cut -d ',' -f 5
```

### (Optional) Run GLUE benchmark as an upper limit

This extension can also evaluate SSL models for language representation with text input (i.e., running GLUE itself) and the results are treated as the performance upper limit of SpeechGLUE.  
Note that the text data of SpeechGLUE is modified from the original by text normalization for TTS, and is therefore also used as it is in this GLUE benchmark for a fair comparison.

```shell
# e.g., evaluate the BERT model with base architecture on CoLA task
TASK=cola
UPSTREAM=bert-base-uncased
OUTDIR=result/${TASK}/${UPSTREAM}

python run_downstream.py \
  -m train \
  -p ${OUTDIR} \
  -u hf_nlp_ssl \
  --upstream_ckpt ${UPSTREAM} \
  -d glue \
  -c downstream/glue/config_${TASK}.yaml
# to evaluate on other tasks, specify the task key in the above table
```

You can find other text-based SSL models from [Huggin Face](https://huggingface.co/) (e.g., [bert-large-uncased](https://huggingface.co/bert-large-uncased) and [data2vec-text-base](https://huggingface.co/facebook/data2vec-text-base)).

### (Optional) Run other baselines

In addition to the FBANK baseline, the extension offers other baselines as an upstream component, specifically phoneme input and a randomly initialized model (i.e., without applying SSL).

#### phoneme input model

Since we utilize ESPnet for the TTS system, the grapheme-to-phoneme (G2P) converter is based on the G2P used in ESPnet.

```shell
# e.g., evaluate the phoneme input model on CoLA task
# in the case of adopting "kan-bayashi/ljspeech_vits" for TTS in the data preparation
TASK=cola
UPSTREAM="kan-bayashi/ljspeech_vits"
UPSTREAM_BASE=`basename ${UPSTREAM}`
OUTDIR=result/${TASK}/${UPSTREAM_BASE}_g2p
EMB_SIZE=128

python run_downstream.py \
  -m train \
  -p ${OUTDIR} \
  -u embedding \
  --upstream_trainable \
  --upstream_ckpt ${UPSTREAM} \
  --upstream_model_config upstream/embedding/embedding_size_${EMB_SIZE}.yaml \
  -d glue \
  -c downstream/glue/config_${TASK}_ph.yaml
# to evaluate on other tasks, specify the task key in the above table
```

#### randomly initialized model (i.e., without applying SSL)

```shell
# e.g., evaluate the randomly initialized model with LARGE architecture on CoLA task
TASK=cola
UPSTREAM="facebook/wav2vec2-large-lv60"
UPSTREAM_BASE=`basename ${UPSTREAM}`
OUTDIR=result/${TASK}/${UPSTREAM_BASE}_no_pretrained_weights

python run_downstream.py \
  -m train \
  -p ${OUTDIR} \
  -u hf_speechssl_no_pretrained_weights \
  --upstream_ckpt ${UPSTREAM} \
  -d speechglue \
  -c downstream/speechglue/config_${TASK}.yaml
# to evaluate on other tasks, specify the task key in the above table
```

You can also evaluate the base architecture w/o SSL by specifying `facebook/wav2vec2-base` for `UPSTREAM`.

### (Optional) Run ASR tasks on the speech data of SpeechGLUE

In order to validate the quality of synthesized speech, speech SSL models can further address the ASR tasks on the speech data of SpeechGLUE.
First, generate a dictionary for each task:

```shell
# e.g., generate the dictionary for CoLA task
TASK=cola
python downstream/speechglue_asr/mk_char_dict.py --glue-task $TASK
```

Then, for high-resource tasks, since the amount of data is too large to check the quality of TTS outputs, we randomly select a maximum of 100 hours of data from the training set (as in SUPERB, where only *train-clean-100* from LibriSpeech is utilized):

```shell
python downstream/speechglue_asr/select_sample.py --split train --max-hours 100
# also randomly select the development and test sets to be a maximum of 5 hours 
python downstream/speechglue_asr/select_sample.py --split validation --max-hours 5
python downstream/speechglue_asr/select_sample.py --split test --max-hours 5
```

Finally, run ASR tasks:

```shell
# e.g., evaluate the HuBERT model with base architecture on CoLA dataset
TASK=cola
UPSTREAM=hubert_base
OUTDIR=result/${TASK}_asr/${UPSTREAM}

python run_downstream.py \
  -m train \
  -p ${OUTDIR} \
  -u ${UPSTREAM} \
  -d speechglue_asr \
  -c downstream/speechglue_asr/config_${TASK}.yaml
# to evaluate on other tasks, specify the task key in the above table

# additionally, to evaluate on a test set, execute the following command
python run_downstream.py \
  -m evaluate \
  -t "test" \
  -e ${OUTDIR}/dev-best.ckpt
```

## Citation

If you find this helpful, please consider citing the following paper (to be published).

```text
@inproceedings{ashihara23_interspeech,
  author={Takanori Ashihara and Takafumi Moriya and Kohei Matsuura and Tomohiro Tanaka and Yusuke Ijima and Taichi Asami and Marc Delcroix and Yukinori Honma},
  title={{SpeechGLUE: How} Well Can Self-Supervised Speech Models Capture Linguistic Knowledge?},
  year=2023,
  booktitle={Proc. Interspeech 2023}
}
```

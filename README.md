# Berserker
Berserker (BERt chineSE woRd toKenizER) is a Chinese tokenizer built on top of Google's [BERT](https://github.com/google-research/bert) model.

## Installation
```python
pip install basaka
```

## Usage
```python
import berserker

berserker.load_model() # An one-off installation
berserker.tokenize('姑姑想過過過兒過過的生活。') # ['姑姑', '想', '過', '過', '過兒', '過過', '的', '生活', '。']
```

## Benchmark
The table below shows that Berserker achieved state-of-the-art F1 measure on the [SIGHAN 2005](http://sighan.cs.uchicago.edu/bakeoff2005/) [dataset](http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip).

The result below is trained with 15 epoches on each dataset with a batch size of 64.

|                    | PKU      | CITYU    | MSR      | AS       |
|--------------------|----------|----------|----------|----------|
| Liu et al. (2016)  | **96.8** | --       | 97.3     | --       |
| Yang et al. (2017) | 96.3     | 96.9     | 97.5     | 95.7     |
| Zhou et al. (2017) | 96.0     | --       | 97.8     | --       |
| Cai et al. (2017)  | 95.8     | 95.6     | 97.1     | --       |
| Chen et al. (2017) | 94.3     | 95.6     | 96.0     | 94.6     |
| Wang and Xu (2017) | 96.5     | --       | 98.0     | --       |
| Ma et al. (2018)   | 96.1     | **97.2** | 98.1     | 96.2     |
|--------------------|----------|----------|----------|----------|
| Berserker          | 96.6     | 97.1     | **98.4** | **96.5** |

Reference: [Ji Ma, Kuzman Ganchev, David Weiss - State-of-the-art Chinese Word Segmentation with Bi-LSTMs](https://arxiv.org/pdf/1808.06511.pdf)

## Limitation
Since Berserker ~~is muscular~~ is based on BERT, it has a large model size (~300MB) and run slowly on CPU. Berserker is just a proof of concept on what could be achieved with BERT.

Currently the default model provided is trained with [SIGHAN 2005](http://sighan.cs.uchicago.edu/bakeoff2005/) [PKU dataset](http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip). We plan to release more pretrained model in the future.

## Architecture
Berserker is fine-tuned over TPU with [pretrained Chinese BERT model](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip). It is connected with a single dense layer which is applied to all tokens to produce a sequence of [0, 1] output, where 1 denote a split.

## Training
We provided the source code for training under the `trainer` subdirectory. Feel free to contact me if you need any help reproducing the result.

## Bonus Video
[<img src="https://img.youtube.com/vi/H_xmyvABZnE/maxres1.jpg" alt="Yachae!! BERSERKER!!"/>](https://www.youtube.com/watch?v=H_xmyvABZnE)

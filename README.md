# Berserker
Berserker (BERt chineSE toKenizER) is a Chinese tokenizer built on top of Google's [BERT](https://github.com/google-research/bert) model.

## Installation
```python
pip install berserker
```

## Usage
```python
import berserker

berserker.load_model('pku')
berserker.tokenize('姑姑想過過過兒過過的生活。') # ... (By defaul use PKU dataset trained tokenizer)
```

## Training
Berserker is fine-tuned over TPU with [pretrained Chinese BERT model](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip). It is connected with a single dense layer which is applied to all tokens to produce a sequence of [0, 1] output.

## Result
A quick test shows that Berserker achieved F1 measure of 0.965 on the [SIGHAN 2005](http://sighan.cs.uchicago.edu/bakeoff2005/) PKU [dataset](http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip). This is trained by using the full training data and trained for 5000 steps with batch size 64.

```
=== SUMMARY:
=== TOTAL INSERTIONS:	727
=== TOTAL DELETIONS:	1800
=== TOTAL SUBSTITUTIONS:	2413
=== TOTAL NCHANGE:	4940
=== TOTAL TRUE WORD COUNT:	104372
=== TOTAL TEST WORD COUNT:	103299
=== TOTAL TRUE WORDS RECALL:	0.960
=== TOTAL TEST WORDS PRECISION:	0.970
=== F MEASURE:	0.965
=== OOV Rate:	0.058
=== OOV Recall Rate:	0.851
=== IV Recall Rate:	0.966
```

more to come...

## Bonus Video
[<img src="https://img.youtube.com/vi/H_xmyvABZnE/maxres1.jpg" alt="Yachae!! BERSERKER!!"/>](https://www.youtube.com/watch?v=H_xmyvABZnE)

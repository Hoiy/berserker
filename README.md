# Berserker
Berserker (BERt chineSE toKenizER) is a Chinese tokenizer built on top of Google's [BERT](https://github.com/google-research/bert) model.

## Training
Berserker is fine-tuned over TPU with [pretrained Chinese BERT model](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip). It is connected with a single dense layer which is applied to all tokens to produce a sequence of [0, 1] output.

## Result
A quick test shows that Berserker achieved F1 measure of 0.960 on the [SIGHAN 2005](http://sighan.cs.uchicago.edu/bakeoff2005/) PKU [dataset](http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip).

```
=== SUMMARY:
=== TOTAL INSERTIONS:	781
=== TOTAL DELETIONS:	2159
=== TOTAL SUBSTITUTIONS:	2723
=== TOTAL NCHANGE:	5663
=== TOTAL TRUE WORD COUNT:	104372
=== TOTAL TEST WORD COUNT:	102994
=== TOTAL TRUE WORDS RECALL:	0.953
=== TOTAL TEST WORDS PRECISION:	0.966
=== F MEASURE:	0.960
=== OOV Rate:	0.058
=== OOV Recall Rate:	0.855
=== IV Recall Rate:	0.959
```

more to come...

## Bonus Video
[<img src="https://img.youtube.com/vi/H_xmyvABZnE/maxres1.jpg" alt="Yachae!! BERSERKER!!"/>](https://www.youtube.com/watch?v=H_xmyvABZnE)

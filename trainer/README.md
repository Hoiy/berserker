# Training

First, clone the Berserker repository.

All the command below should be run within `berserker` directory and **not** the `trainer` subdirectory.

## Download assets
The command below download to the `assets` subdirectory

1. [SIGHAN 2005 dataset](http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip).
2. [Google BERT pretrained Chinese model](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip).

```sh
python -m trainer.assets
```

## Prepare dataset
The command below convert all training data from `./assets` into tfrecords for training.
```sh
python -O -m trainer.dataset
```

## Training and Prediction (over TPU)
It is suggest to train the model over TPU.

One option is to use Colab with free TPU support. Make sure you have Google Cloud Platform access. You need to upload all the dependencies into Google Storage, e.g. the full Berserker repository, generated tfrecords, test files, etc... It is recommend to use `gsutil rsync` to synchronize everything into Google Storage.

In Colab notebook, execute the following command after obtaining a TPU address.

```sh
!python -m trainer.task \
  --use_tpu=true \
  --tpu_name={TPU_ADDRESS} \
  --checkpoint_dir=gs://path/to/model/checkpoint \
  --train_file=gs://path/to/dataset.tfrecords \
  --predict_file=gs://path/to/prediction/input/file/for/example/pku_testing.utf8 \
  --predict_output=gs://path/to/prediction/output/file/for/example/pku_pred.utf8 \
  --train_steps={train_steps} \
  --do_train=true \
  --do_predict=true \
  --do_export=true \
  --output_dir=gs://path/to/export/trained/model
```  

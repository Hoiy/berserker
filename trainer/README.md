# Training

Under construction...

First, clone this repository

## Download assets
```sh
python -m trainer.assets
```

## Prepare dataset
```sh
python -O -m trainer.dataset
```

## Verify dataset
```sh
python -m trainer.view_tfrecords
```

## Training (over TPU)
```sh
python run.py \
  --use_tpu=true \
  --do_train=true
```

## Training (in Local)
```sh
python -m trainer.task \
  --use_tpu=false \
  --do_train=true
```


## Prediction (over CPU)
```sh
python -m trainer.task \
  --use_tpu=false \
  --do_predict=true
  --predict_file=<text_file>
```

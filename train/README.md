# Training

Under construction...

## Download assets
```sh
python assets.py
```

## Prepare dataset
```sh
python dataset.py
```

## Training (over TPU)
```sh
python run.py \
  --use_tpu=true \
  --do_train=true
```

## Prediction
```sh
python run.py \
  --use_tpu=true \
  --do_predict=true
  --predict_file=<text_file>
```

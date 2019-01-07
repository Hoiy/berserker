## Prepare dataset
`PYTHONPATH=$(pwd) python train/prepare_dataset.py --output_dir=gs://<bucket>/dataset`

## Eval
`PYTHONPATH=$(pwd) python train/eval.py --model_dir=<model_dir>`

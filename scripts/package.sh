#!/bin/bash
set -e

MODEL=$1
if [[ $MODEL == 'pku' ]]; then MODEL_CODE=1550302575
elif [[ $MODEL == 'cityu' ]]; then MODEL_CODE=1550311590
elif [[ $MODEL == 'msr' ]]; then MODEL_CODE=1550342892
else
  echo "Usage: package.sh <pku|cityu|msr>"
  exit 1
fi

mkdir -p ./package/${MODEL_CODE}
gsutil -m rsync -rd gs://berserker/${MODEL}_export/${MODEL_CODE} ./package/${MODEL_CODE}
pushd ./package && zip -r ${MODEL_CODE}.zip ${MODEL_CODE} && popd

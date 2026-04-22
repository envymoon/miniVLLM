#!/bin/bash

if [ -z "$1" ]; then
    echo "instrction: ./run_kernel.sh <kernel_file.cu>"
    exit 1
fi

FULL_PATH=$1
FILE_NAME=$(basename -- "$FULL_PATH")
BASE_NAME="${FILE_NAME%.*}"

mkdir -p bench/bin bench/logs

make "bench/bin/$BASE_NAME" > "bench/logs/${BASE_NAME}_build.log" 2>&1

if [ $? -ne 0 ]; then
    echo "failed compiling bench/logs/${BASE_NAME}_build.log"
    exit 1
fi

"./bench/bin/$BASE_NAME" > "bench/logs/${BASE_NAME}_run.txt"

ncu --csv --log-file "bench/logs/${BASE_NAME}_ncu.csv" "./bench/bin/$BASE_NAME"
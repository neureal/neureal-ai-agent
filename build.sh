#!/usr/bin/bash

set -eu

CWD=$(basename "$PWD")
export DEV=1

build() {
    docker build . --tag "$CWD"
}

buildpriv() {
    export PRIV=1
    docker build . --tag "$CWD" --build-arg PRIV
}

clean() {
    docker system prune -f
}

dev() {
    # TF_* vars: https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#tensorcore
    mkdir -p output
    mkdir -p logs
    mkdir -p tf-data-models-local
    mkdir -p tf-data-models
    docker run --rm --gpus=all \
        -e DEV \
        -e MT5IP \
        -e TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=1 \
        -e TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=1 \
        -e TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=1 \
        -v "$PWD"/output:/app/output \
        -v "$PWD"/logs:/app/logs \
        -v "$PWD"/tf-data-models-local:/root/tf-data-models-local \
        -v "$PWD"/tf-data-models:/root/tf-data-models \
        -v "$PWD"/..:/outerdir \
	-p 6006:6006 \
        -it "$CWD" "$@"
}

run() {
    export DEV=
    dev "$@"
}

case ${1:-build} in
    build) build ;;
    buildpriv) buildpriv ;;
    clean) clean ;;
    dev) dev "$@" ;;
    run) run "$@" ;;
    *) echo "$0: No command named '$1'" ;;
esac

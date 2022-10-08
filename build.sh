#!/bin/sh

set -eu

CWD=$(basename "$PWD")
export DEV=1
export MT5IP

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
    mkdir -p output
    mkdir -p tf-data-models-local
    mkdir -p tf-data-models
    docker run --rm --gpus=all \
        -e DEV \
        -e MT5IP \
        -v "$PWD"/output:/app/output \
        -v "$PWD"/tf-data-models-local:/root/tf-data-models-local \
        -v "$PWD"/tf-data-models:/root/tf-data-models \
        -v "$PWD"/..:/outerdir \
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

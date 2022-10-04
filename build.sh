#!/bin/sh

set -eu

CWD=$(basename "$PWD")

build() {
    docker build . --tag "$CWD"
}

clean() {
    docker system prune -f
}

dev() {
    mkdir -p output
    mkdir -p tf-data-models-local
    docker run --rm --gpus=all --entrypoint=bash \
        -v "$PWD"/output:/app/output \
        -v "$PWD"/tf-data-models-local:/app/tf-data-models-local \
        -p 8080:8080 \
        -it "$CWD"
}

run() {
    shift
    mkdir -p output
    mkdir -p tf-data-models-local
    docker run --rm --gpus=all \
        -v "$PWD"/output:/app/output \
        -v "$PWD"/tf-data-models-local:/app/tf-data-models-local \
        "$CWD" "$@"
}

case ${1:-build} in
    build) build ;;
    clean) clean ;;
    dev) dev "$@" ;;
    run) run "$@" ;;
    *) echo "$0: No command named '$1'" ;;
esac

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
#    mkdir -p output
#        -v "$PWD"/output:/home/huggingface/output \
    docker run --rm --gpus=all --entrypoint=bash \
        -it "$CWD"
}

run() {
    shift
#    mkdir -p output
#        -v "$PWD"/output:/home/huggingface/output \
    docker run --rm --gpus=all \
        "$CWD" "$@"
}

case ${1:-build} in
    build) build ;;
    clean) clean ;;
    dev) dev "$@" ;;
    run) run "$@" ;;
    *) echo "$0: No command named '$1'" ;;
esac

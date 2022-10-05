#!/bin/sh

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
    mkdir -p output
    mkdir -p tf-data-models-local
    docker run --rm --gpus=all \
	-e DEV \
        -v "$PWD"/output:/app/output \
        -v "$PWD"/tf-data-models-local:/app/tf-data-models-local \
	-v "$PWD"/..:/outerdir \
        -p 8080:8080 \
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

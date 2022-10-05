# Stage 1: TA-Lib Builder
FROM ubuntu:20.04 as talib-builder
RUN apt-get update && \
  apt-get install --no-install-recommends -y wget build-essential file
# Instructions from: https://mrjbq7.github.io/ta-lib/install.html
RUN mkdir /ta-lib
WORKDIR /ta-lib
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
RUN tar -xvzf ta-lib-0.4.0-src.tar.gz
WORKDIR /ta-lib/ta-lib
RUN ./configure --prefix=/usr
RUN make && make install

# Stage 2: Python Builder
FROM python:3.8-slim-bullseye as python-builder
ENV DEBIAN_FRONTEND noninteractive
ENV PATH="${PATH}:/root/.local/bin"
RUN apt-get update && \
  apt-get install --no-install-recommends -y build-essential gcc git python3-pip
COPY requirements.txt /
COPY --from=talib-builder /usr/lib/libta_lib.so.0.0.0 /usr/lib/
RUN cd /usr/lib && ln -s libta_lib.so.0.0.0 libta_lib.so.0
RUN cd /usr/lib && ln -s libta_lib.so.0.0.0 libta_lib.so
COPY --from=talib-builder /usr/lib/libta_lib.la /usr/lib/
COPY --from=talib-builder /usr/lib/libta_lib.a /usr/lib/
COPY --from=talib-builder /usr/bin/ta-lib-config /usr/bin/ta-lib-config
COPY --from=talib-builder /usr/include/ta-lib /usr/include/ta-lib
RUN ldconfig

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --user -r /requirements.txt

# Stage 3: Runtime
# CUDA 11.2.2_461.33, CUDNN 8.1.1.33, tensorflow-gpu==2.9.1, tensorflow_probability==0.17.0
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV TZ="America/Denver"
ENV DEBIAN_FRONTEND noninteractive
ENV PATH="${PATH}:/root/.local/bin"
ARG PRIV
ENV PRIV="${PRIV}"
RUN apt-get update && \
  apt-get install --no-install-recommends -y build-essential gcc git wget curl ca-certificates vim openssh-client \
  python3.8 python3-dev python3-pip python3-distutils python3-venv pylint3 && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=talib-builder /usr/lib/libta_lib.so.0.0.0 /usr/lib/
RUN cd /usr/lib && ln -s libta_lib.so.0.0.0 libta_lib.so.0
RUN cd /usr/lib && ln -s libta_lib.so.0.0.0 libta_lib.so
COPY --from=talib-builder /usr/lib/libta_lib.la /usr/lib/
COPY --from=talib-builder /usr/lib/libta_lib.a /usr/lib/
COPY --from=talib-builder /usr/bin/ta-lib-config /usr/bin/ta-lib-config
COPY --from=talib-builder /usr/include/ta-lib /usr/include/ta-lib
RUN ldconfig
COPY --from=python-builder /root/.local/lib/python3.8/site-packages /usr/local/lib/python3.8/dist-packages

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . /app
WORKDIR /app

COPY entrypoint.sh /usr/bin/entrypoint
ENTRYPOINT ["/usr/bin/entrypoint"]

# Stage 1: Builder/Compiler
FROM python:3.8-slim-bullseye as builder

ENV DEBIAN_FRONTEND noninteractive
ENV PATH="${PATH}:/root/.local/bin"
RUN apt-get update && \
  apt-get install --no-install-recommends -y build-essential gcc git \
  python3-pip python3-matplotlib python3-numba python3-numpy
COPY requirements.txt /requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --user -r /requirements.txt


# Stage 2: Runtime
# CUDA 11.2.2_461.33, CUDNN 8.1.1.33, tensorflow-gpu==2.9.1, tensorflow_probability==0.17.0
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV TZ="America/Denver"
ENV DEBIAN_FRONTEND noninteractive
ENV PATH="${PATH}:/root/.local/bin"
RUN apt-get update && \
  apt-get install --no-install-recommends -y build-essential gcc git wget curl ca-certificates \
  python3.8 python3-dev python3-pip python3-distutils && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local/lib/python3.8/site-packages /usr/local/lib/python3.8/dist-packages

# TA-Lib Dependency
# https://mrjbq7.github.io/ta-lib/install.html
RUN mkdir /ta-lib
WORKDIR /ta-lib
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
RUN tar -xvzf ta-lib-0.4.0-src.tar.gz
WORKDIR /ta-lib/ta-lib
RUN ./configure --prefix=/usr
RUN make && make install
# pip TA-Lib
RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install TA-Lib==0.4.25

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . /app
WORKDIR /app
ENV PYTHONPATH="${PYTHONPATH}:/app/neureal-ai-util/"

CMD ["python", "/app/app.py"]

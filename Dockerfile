FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata
RUN apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python3-setuptools \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    iptables \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /pill-identification-server
RUN git clone https://github.com/leeyeoreum02/pill-identification.git /pill-identification-server
RUN pip install -r requirements.txt

RUN pip install gdown
RUN mkdir /pill-identification-server/weights
RUN gdown --fuzzy https://drive.google.com/file/d/13hkj0zLKpBb-lxJ2yrPZLPARfidfBbM8/view?usp=share_link
RUN unzip /pill-identification-server/weights.zip -d /pill-identification-server/weights \
    && rm /pill-identification-server/weights.zip

EXPOSE 3636

CMD uvicorn app.main:app --host 0.0.0.0 --port 3636

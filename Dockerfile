FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /pill-identification-server
RUN git clone https://github.com/leeyeoreum02/pill-identification.git /pill-identification-server
RUN pip install -r requirements.txt

RUN pip install gdown
RUN mkdir /pill-identification-server/data /pill-identification-server/weights
RUN gdown -O /pill-identification-server/data/OpenData_PotOpenTabletIdntfc20230319.xls \
    --fuzzy https://docs.google.com/spreadsheets/d/13qsKWSgpOyVj-J9Voro9p7IWTbUlBksp/edit?usp=sharing&ouid=103679390798631008004&rtpof=true&sd=true
RUN gdown --fuzzy https://drive.google.com/file/d/13hkj0zLKpBb-lxJ2yrPZLPARfidfBbM8/view?usp=share_link
RUN unzip /pill-identification-server/weights.zip -d /pill-identification-server/weights \
    && rm /pill-identification-server/weights.zip
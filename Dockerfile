FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
ARG BRANCH=main

WORKDIR /root/

RUN apt-get update && apt-get install -y git wget

RUN git clone --branch $BRANCH https://github.com/mkrum/stonefish.git && \
            cd stonefish && \
            pip install -r requirements.txt && \
            pip install -e . && \
            cd -


WORKDIR /root/stonefish

RUN wget https://www.dropbox.com/s/4xyyhue8u49m7pz/test_5.csv?dl=0 -O ./data/test_5.csv 
RUN wget https://www.dropbox.com/s/h64v4f39qn4vo6v/train_5.csv?dl=0 -O ./data/train_5.csv

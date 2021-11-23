FROM pytorch/pytorch:latest

WORKDIR /root/

RUN git clone https://github.com/mkrum/stonefish.git && \
            cd stonefish && \
            pip install -r requirements.txt && \
            pip install -e .

WORKDIR /root/stonefish

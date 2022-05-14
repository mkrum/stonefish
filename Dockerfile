FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt update && apt install -y python3.8 python3-pip build-essential git cmake clang-9 clang-tools-9 git wget

ENV LD_LIBRARY_PATH /usr/local/lib

RUN mkdir /workdir                                                                              
                                                                                                
RUN git clone https://github.com/fogleman/MisterQueen.git /workdir/MisterQueen                  
RUN cd /workdir/MisterQueen && make COMPILE_FLAGS="-std=c99 -Wall -O3 -fPIC" && cd ..
RUN gcc -shared -o /usr/local/lib/libmisterqueen.so /workdir/MisterQueen/build/release/*.o -lpthread
RUN gcc -shared -o /usr/local/lib/libtinycthread.so /workdir/MisterQueen/build/release/deps/tinycthread/tinycthread.o -lpthread

RUN apt-get install -y python3.8-dev libffi-dev
RUN python3.8 -m pip install cython 
RUN python3.8 -m pip install cffi numpy

WORKDIR /workdir/
RUN git clone https://github.com/mkrum/chessenv.git 
RUN mv MisterQueen chessenv/
RUN cd chessenv && pip install -e . && cd ..


RUN git clone https://github.com/official-stockfish/Stockfish.git && cd Stockfish/src/ &&  make net && make build ARCH=x86-64-modern && cd

ENV PATH $PATH:/workdir/Stockfish/src

RUN git clone https://github.com/mkrum/stonefish.git /stonefish && cd /stonefish && pip install -r requirements.txt && pip install -e .

WORKDIR /stonefish

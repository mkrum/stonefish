FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install base dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    wget \
    libffi-dev \
    clang \
    libc++-dev \
    libc++abi-dev

ENV LD_LIBRARY_PATH=/usr/local/lib

RUN pip install uv

WORKDIR /workdir

# Setup chess environment
RUN git clone https://github.com/mkrum/chessenv.git /opt/chessenv && cd /opt/chessenv && ./build_lib.sh && python copy_libs.py  && uv build && uv sync && uv pip install --system . && cd /workdir

RUN git clone --depth 1 https://github.com/fogleman/MisterQueen.git /workdir/MisterQueen
RUN cd /workdir/MisterQueen && make COMPILE_FLAGS="-std=c99 -Wall -O3 -fPIC" && cd ..
RUN gcc -shared -o /usr/local/lib/libmisterqueen.so /workdir/MisterQueen/build/release/*.o -lpthread
RUN gcc -shared -o /usr/local/lib/libtinycthread.so /workdir/MisterQueen/build/release/deps/tinycthread/tinycthread.o -lpthread

# Setup Stockfish
RUN git clone https://github.com/official-stockfish/Stockfish.git && \
    cd Stockfish/src/ && make build net && cd

ENV PATH=$PATH:/workdir/Stockfish/src

# Install Python dependencies (CPU-only version)
WORKDIR /stonefish

COPY requirements.txt /stonefish/
RUN uv pip install --system --no-cache-dir -r requirements.txt --index-url https://pypi.org/simple/ --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple/

# Install stonefish
COPY . /stonefish
RUN uv pip install --system -e .

# Set clang as the default compiler
ENV CC=clang
ENV CXX=clang++

RUN uv pip install --system open_spiel
# Set default command
CMD ["bash"]

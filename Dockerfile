# Use multi-platform PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime


# Install base dependencies
RUN apt update && apt install -y \
    build-essential \
    git \
    cmake \
    clang \
    wget \
    python3-dev \
    libffi-dev

ENV LD_LIBRARY_PATH /usr/local/lib

# Setup MisterQueen
WORKDIR /workdir
RUN git clone --depth 1 https://github.com/fogleman/MisterQueen.git /workdir/MisterQueen                  
RUN cd /workdir/MisterQueen && make COMPILE_FLAGS="-std=c99 -Wall -O3 -fPIC" && cd ..
RUN gcc -shared -o /usr/local/lib/libmisterqueen.so /workdir/MisterQueen/build/release/*.o -lpthread
RUN gcc -shared -o /usr/local/lib/libtinycthread.so /workdir/MisterQueen/build/release/deps/tinycthread/tinycthread.o -lpthread

# Python package dependencies
RUN pip install --upgrade pip 
RUN pip install wheel cython
RUN pip install --no-cache-dir cffi numpy

# Setup chess environment
RUN git clone https://github.com/mkrum/chessenv.git 
RUN mv MisterQueen chessenv/
RUN cd chessenv && pip install -e . && cd ..

# Setup Stockfish with architecture detection
RUN git clone https://github.com/official-stockfish/Stockfish.git && cd Stockfish/src/ && make net && \
    ([ "$(uname -m)" = "arm64" ] || [ "$(uname -m)" = "aarch64" ] && make build ARCH=apple-silicon || make build ARCH=x86-64-modern) && cd
ENV PATH $PATH:/workdir/Stockfish/src

# Install dependencies first for better caching
WORKDIR /stonefish
COPY requirements.lock /stonefish/
RUN pip install --no-cache-dir -r requirements.lock

# Then install stonefish
COPY . /stonefish
RUN pip install -e . --no-deps

# Set default command
CMD ["bash"]

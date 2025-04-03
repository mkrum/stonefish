# CPU-only Dockerfile for stonefish (Default)
FROM python:3.10-slim

# Install base dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    wget \
    libffi-dev

ENV LD_LIBRARY_PATH=/usr/local/lib

# Setup MisterQueen
WORKDIR /workdir
RUN git clone --depth 1 https://github.com/fogleman/MisterQueen.git /workdir/MisterQueen
RUN cd /workdir/MisterQueen && make COMPILE_FLAGS="-std=c99 -Wall -O3 -fPIC" && cd ..
RUN gcc -shared -o /usr/local/lib/libmisterqueen.so /workdir/MisterQueen/build/release/*.o -lpthread
RUN gcc -shared -o /usr/local/lib/libtinycthread.so /workdir/MisterQueen/build/release/deps/tinycthread/tinycthread.o -lpthread

# Setup chess environment
RUN git clone https://github.com/mkrum/chessenv.git
RUN mv MisterQueen chessenv/
RUN cd chessenv && pip install -e . && cd ..

# Setup Stockfish with macOS architecture detection
RUN git clone https://github.com/official-stockfish/Stockfish.git && \
    cd Stockfish/src/ && make build net && cd
ENV PATH=$PATH:/workdir/Stockfish/src

# Install Python dependencies (CPU-only version)
WORKDIR /stonefish
COPY requirements-cpu.txt /stonefish/
RUN pip install --no-cache-dir -r requirements-cpu.txt
# Manually install PyTorch CPU version to avoid CUDA issues
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install stonefish
COPY . /stonefish
RUN pip install -e .

# Set default command
CMD ["bash"]

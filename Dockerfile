# Multi-stage build for smaller image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    wget \
    libffi-dev \
    clang \
    libc++-dev \
    libc++abi-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /build

# Build chess environment
RUN git clone https://github.com/mkrum/chessenv.git /build/chessenv && \
    cd /build/chessenv && \
    ./build_lib.sh && \
    python copy_libs.py && \
    uv pip install --system . && \
    ls -la /usr/local/lib/

# Build MisterQueen
RUN git clone --depth 1 https://github.com/fogleman/MisterQueen.git /build/MisterQueen && \
    cd /build/MisterQueen && \
    make COMPILE_FLAGS="-std=c99 -Wall -O3 -fPIC" && \
    gcc -shared -o /build/libmisterqueen.so build/release/*.o -lpthread && \
    gcc -shared -o /build/libtinycthread.so build/release/deps/tinycthread/tinycthread.o -lpthread

# Build Stockfish
RUN git clone --depth 1 https://github.com/official-stockfish/Stockfish.git /build/Stockfish && \
    cd /build/Stockfish/src/ && \
    make build net

# Final stage - needs devel for torch.compile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PATH=$PATH:/usr/local/bin/stockfish

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install --no-cache-dir uv

# Copy built artifacts from builder
COPY --from=builder /build/libmisterqueen.so /usr/local/lib/
COPY --from=builder /build/libtinycthread.so /usr/local/lib/
COPY --from=builder /build/Stockfish/src/stockfish /usr/local/bin/stockfish/
# Copy installed fastchessenv Python package (includes the libraries in fastchessenv/lib/)
COPY --from=builder /opt/conda/lib/python3.10/site-packages/fastchessenv /opt/conda/lib/python3.10/site-packages/fastchessenv
COPY --from=builder /opt/conda/lib/python3.10/site-packages/fastchessenv*.dist-info /opt/conda/lib/python3.10/site-packages/
# Copy the compiled C extension module
COPY --from=builder /opt/conda/lib/python3.10/site-packages/fastchessenv_c*.so /opt/conda/lib/python3.10/site-packages/

WORKDIR /stonefish

# Install Python dependencies first (these change less frequently)
# This layer will be cached unless requirements.txt changes
COPY requirements.txt /stonefish/
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy only setup files for installing the package
# This layer will be cached unless setup.py/pyproject.toml changes
COPY setup.py pyproject.toml readme.md /stonefish/
# Create package structure needed for installation
RUN mkdir -p /stonefish/stonefish/eval /stonefish/stonefish/train
COPY stonefish/__init__.py /stonefish/stonefish/
COPY stonefish/eval/__init__.py /stonefish/stonefish/eval/
COPY stonefish/train/__init__.py /stonefish/stonefish/train/
RUN uv pip install --system --no-deps -e .

# Copy the rest of the application code last
# Only this layer rebuilds when you change code
COPY . /stonefish

CMD ["bash"]

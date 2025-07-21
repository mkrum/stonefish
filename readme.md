# 🪨🐟 stonefish: Transformers (and more) For Chess

This repository contains code for training, evaluating, and deploying deep learning models for playing chess.

- [x] beat random
- [x] beat stockfish at depth 1
- [x] beat myself [(write up)](https://approximatemethods.com/chess.html)
- [ ] beat someone good
- [ ] beat everyone

Search algorithms are considered cheating.

## Quick Start

```bash
# Train locally
just train-local configs/train_resnet_small.yml
```
The code can also be deployed to a [GKE cluster](https://cloud.google.com/kubernetes-engine).
```bash
# Deploy to k8s
just up
just train-deploy
```

See the [justfile](justfile) for more details.

## Setup

```bash
# Clone and install
git clone https://github.com/mkrum/stonefish.git
cd stonefish
pip install -e .[dev,test]
```
There is also a dockerfile, which you can build with:
```bash
just build-local
```

## Training

```bash
# Local training
just train-local configs/train_resnet_big.yml

# Local with custom parameters
python -m stonefish.train configs/train_convnet_small.yml output_dir

# Distributed training (4 GPUs)
torchrun --standalone --nproc_per_node=4 -m stonefish.train configs/train_resnet_big.yml output_dir
```

## Evaluation

```bash
# Model vs Random
python -m stonefish.eval --agent1 model:configs/train_resnet_big.yml:checkpoint.pth --agent2 random --games 100

# Model vs Stockfish
python -m stonefish.eval --agent1 model:configs/train_convnet_big.yml:model.pth --agent2 stockfish:depth=3 --games 50

# Model vs Model
python -m stonefish.eval --agent1 model:config1.yml:model1.pth --agent2 model:config2.yml:model2.pth --games 200
```

## Benchmarking

```bash
# Benchmark model throughput
just benchmark configs/train_resnet_big.yml

# With specific batch sizes
python stonefish/benchmark.py configs/train_convnet_big.yml --batch-sizes 1 8 32 128 --device cuda
```

## Kubernetes

```bash
# Full deployment
just up                  # Create cluster
just train-deploy        # Start training
kubectl logs -f job/stonefish-training  # Watch logs
just cleanup            # Tear down

# Debug shell
just shell
```

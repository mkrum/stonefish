[![Build Status](https://app.travis-ci.com/mkrum/stonefish.svg?branch=main)](https://app.travis-ci.com/mkrum/stonefish)
```
 ____ _____ ___  _   _ _____ _____ ___ ____  _   _ 
/ ___|_   _/ _ \| \ | | ____|  ___|_ _/ ___|| | | |
\___ \ | || | | |  \| |  _| | |_   | |\___ \| |_| |
 ___) || || |_| | |\  | |___|  _|  | | ___) |  _  |
|____/ |_| \___/|_| \_|_____|_|   |___|____/|_| |_|
```

# Quickstart
There are two docker images available mkrum/stonefish (Dockerfile),
mkrum/stonefish-final (Dockerfile-final). The final version includes the
t5-large and t5-base models. 

# How To Run
```
docker run -it --rm mkrum/stonefish python stonefish/train/t5.py t5-base
--batch_size 155
```
```
docker run -it --rm mkrum/stonefish python stonefish/train/rl_lm.py t5-base
<path> 
```

# About
There are two main files for review, `stonefish/train/t5.py`, which contains the
standard fine-tuning code, and `stonefish/train/rl_lm.py`, which contains the rl
finetuning step.

# Installation
You can install via pip.
```
pip install -r requirements.txt
pip install -e .
```

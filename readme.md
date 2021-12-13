[![Build Status](https://app.travis-ci.com/mkrum/stonefish.svg?branch=main)](https://app.travis-ci.com/mkrum/stonefish)
```
 ____ _____ ___  _   _ _____ _____ ___ ____  _   _ 
/ ___|_   _/ _ \| \ | | ____|  ___|_ _/ ___|| | | |
\___ \ | || | | |  \| |  _| | |_   | |\___ \| |_| |
 ___) || || |_| | |\  | |___|  _|  | | ___) |  _  |
|____/ |_| \___/|_| \_|_____|_|   |___|____/|_| |_|
```

# Quickstart
There are two docker images available from the dockerhub mkrum/stonefish (Dockerfile),
mkrum/stonefish-final (Dockerfile-final). We uploaded our additional data in the
mkrum/stonefish-final image, including our t5-large and t5-base models. These
and information about their training data can be found in the `/nfs/` directory.
We also dockerized the code to run the CommonGen eval. This can be found in the
`CGEval/` folder.

This code is designed to be run with a V100, but should still be able to run
locally (albeit restrictively slow).

# How To Run
```
docker pull mkrum/stonefish
docker pull mkrum/stonefish-final
```

You can try to run the small models as an example:
```
docker run -it --rm mkrum/stonefish python stonefish/train/t5.py t5-small --batch_size 155
```
```
docker run -it --rm mkrum/stonefish python stonefish/train/rl_lm.py t5-small <path_to_weights> 
```
You can also install via pip.
```
pip install -r requirements.txt
pip install -e .
```

# Tip
There are two main files for review, `stonefish/train/t5.py`, which contains the
standard fine-tuning code, and `stonefish/train/rl_lm.py`, which contains the rl
finetuning step. There is a lot of other code in this repository that we didn't
end up using, so I would recommend starting there and moving outwards.


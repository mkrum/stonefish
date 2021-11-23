[![Build Status](https://app.travis-ci.com/mkrum/stonefish.svg?branch=main)](https://app.travis-ci.com/mkrum/stonefish)
```
     _                    __ _     _     
 ___| |_ ___  _ __   ___ / _(_)___| |__  
/ __| __/ _ \| '_ \ / _ \ |_| / __| '_ \ 
\__ \ || (_) | | | |  __/  _| \__ \ | | |
|___/\__\___/|_| |_|\___|_| |_|___/_| |_|
```

# Quickstart
```
docker run -it --rm mkrum/stonefish python -m stonefish.train configs/ttt.yml
```
```
docker run -v $(pwd)/stonefish:/root/stonefish/stonefish -it --rm mkrum/stonefish python -m stonefish.train configs/ttt.yml
```

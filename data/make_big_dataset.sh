#! /bin/bash

touch with_blitz.csv

for year in $(seq 2018 2025); do
    for month in $(seq 6 12); do
	echo $year $month
	HF_HOME=/tmp/hf uv run --env-file .env big_dataset.py $year $month --log-file with_blitz.csv
	rm -rf /tmp/hf
    done
done

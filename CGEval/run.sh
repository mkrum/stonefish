#! /bin/bash

INPUT_FILE=$1
SRC_FILE=${2:-/root/CommonGen-plus/dataset/final_data/commongen/commongen.dev.src_alpha.txt}
TGT_FILE=${3:-/root/CommonGen-plus/dataset/final_data/commongen/commongen.dev.tgt.txt}

cd /CommonGen/methods/unilm_based 

~/anaconda3/envs/unilm/bin/python unilm/src/gigaword/eval.py \
        --pred $INPUT_FILE \
        --gold $TGT_FILE \
        --perl

cd /CommonGen/evaluation/Traditional/eval_metrics/ 

/root/anaconda3/envs/coco_score/bin/python eval.py \
        --key_file $SRC_FILE \
        --gts_file $TGT_FILE \
        --res_file $INPUT_FILE

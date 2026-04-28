#!/usr/bin/env bash
set -e

dataset_name="$1"

if [ -z "$dataset_name" ]; then
    echo "Usage: bash preprocess.sh <UBFC-rPPG|PURE|BUAA>"
    exit 1
fi

if [ "$dataset_name" = "UBFC-rPPG" ]; then
    # UBFC-rPPG
    python script/preprocess.py \
        --dataset_name "UBFC-rPPG" \
        --video_dir "/share2/data/liangqian/UBFC_rPPG/dataset2 " \
        --json_dir "/share2/data/liangqian/UBFC_rPPG/dataset2 " \
        --landmark_dir "/share1/zhaoqiqi/dataset/rPPG/processed/UBFC-rPPG" \
        --h5_dir "/share1/zhaoqiqi/dataset/rPPG/processed/UBFC-rPPG" \
        --store_size 128

elif [ "$dataset_name" = "PURE" ]; then
    # PURE
    python script/preprocess.py \
        --dataset_name "PURE" \
        --video_dir "/share2/data/liangqian/PURE" \
        --landmark_dir "/share1/zhaoqiqi/dataset/rPPG/processed/PURE" \
        --json_dir "/share2/data/liangqian/PURE" \
        --h5_dir "/share1/zhaoqiqi/dataset/rPPG/processed/PURE" \
        --store_size 128

elif [ "$dataset_name" = "BUAA" ]; then
    # # BUAA
    python script/preprocess.py \
        --dataset_name "BUAA" \
        --video_dir "/share2/data/liangqian/BUAA" \
        --landmark_dir "/share1/zhaoqiqi/dataset/rPPG/processed/BUAA" \
        --json_dir "/share2/data/liangqian/BUAA" \
        --h5_dir "/share1/zhaoqiqi/dataset/rPPG/processed/BUAA" \
        --store_size 128

else
    echo "Unsupported dataset_name: $dataset_name"
    echo "Supported: UBFC-rPPG, PURE, BUAA"
    exit 1
fi

#!/bin/bash


## This will clear all images!!
# if [ -d "datasets" ]; then
#     rm -r datasets
# fi

for i in {1..3}; do
    for set in train val; do
        mkdir -p "datasets/fold0$i/$set/images"
        mkdir -p "datasets/fold0$i/$set/labels"
        mkdir -p "datasets/fold0$i/$set/angles"
    done
done

# hard code this shit :)
# 1: (1, 2), 2: (2, 3), 3: (1, 3)
cp -a labelled_data/f01/txt/. "datasets/fold01/train/labels"
cp -a labelled_data/f02/txt/. "datasets/fold01/train/labels"
cp -a labelled_data/f03/txt/. "datasets/fold01/val/labels"

cp -a labelled_data/f02/txt/. "datasets/fold02/train/labels"
cp -a labelled_data/f03/txt/. "datasets/fold02/train/labels"
cp -a labelled_data/f01/txt/. "datasets/fold02/val/labels"

cp -a labelled_data/f01/txt/. "datasets/fold03/train/labels"
cp -a labelled_data/f03/txt/. "datasets/fold03/train/labels"
cp -a labelled_data/f02/txt/. "datasets/fold03/val/labels"


cp -a labelled_data/f01/angle/. "datasets/fold01/train/angles"
cp -a labelled_data/f02/angle/. "datasets/fold01/train/angles"
cp -a labelled_data/f03/angle/. "datasets/fold01/val/angles"

cp -a labelled_data/f02/angle/. "datasets/fold02/train/angles"
cp -a labelled_data/f03/angle/. "datasets/fold02/train/angles"
cp -a labelled_data/f01/angle/. "datasets/fold02/val/angles"

cp -a labelled_data/f01/angle/. "datasets/fold03/train/angles"
cp -a labelled_data/f03/angle/. "datasets/fold03/train/angles"
cp -a labelled_data/f02/angle/. "datasets/fold03/val/angles"




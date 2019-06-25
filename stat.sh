#!/bin/bash

# DROPOUT=0.5
for DROPOUT in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
    echo ""
    echo "Dropout: $DROPOUT"
    echo ""
    for SEED in 1 2 3 4 5 6 7 8 9 10
    do
     ../miniconda3/bin/python twitter.py --no-cuda --epochs=100 --dropout="$DROPOUT" --seed="$SEED"
    done
done

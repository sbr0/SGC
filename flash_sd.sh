#!/bin/bash

ADJ_OFFSET=100
ADJ_SIZE=53060
FEAT_OFFSET=300
FEAT_SIZE=15522256
ADJ_WEIGHTS_OFFSET=50000
ADJ_WEIGHTS_SIZE=53060
CSR_VAL_OFFSET=51000
CSR_VAL_SIZE=196864
CSR_IDX_OFFSET=52000
CSR_IDX_SIZE=196864
CSR_PTR_OFFSET=53000
CSR_PTR_SIZE=10832
LABELS_OFFSET=54000
LABELS_SIZE=10832
PYTHON_STARTING_WEIGHTS_OFFSET=55000
PYTHON_STARTING_WEIGHTS_SIZE=40124
PYTHON_STARTING_BIASES_OFFSET=56000
PYTHON_STARTING_BIASES_SIZE=28

dd if=adj.bin of=/dev/mmcblk0 ibs=512 seek=$ADJ_OFFSET
dd if=features.bin of=/dev/mmcblk0 ibs=512 seek=$FEAT_OFFSET
dd if=adj_weights.bin of=/dev/mmcblk0 ibs=512 seek=$ADJ_WEIGHTS_OFFSET
dd if=CSR_values.bin of=/dev/mmcblk0 ibs=512 seek=$CSR_VAL_OFFSET
dd if=CSR_Idx.bin of=/dev/mmcblk0 ibs=512 seek=$CSR_IDX_OFFSET
dd if=CSR_Ptr.bin of=/dev/mmcblk0 ibs=512 seek=$CSR_PTR_OFFSET
dd if=labels.bin of=/dev/mmcblk0 ibs=512 seek=$LABELS_OFFSET
dd if=python_starting_weights.bin of=/dev/mmcblk0 ibs=512 seek=$PYTHON_STARTING_WEIGHTS_OFFSET
dd if=python_starting_biases.bin of=/dev/mmcblk0 ibs=512 seek=$PYTHON_STARTING_BIASES_OFFSET

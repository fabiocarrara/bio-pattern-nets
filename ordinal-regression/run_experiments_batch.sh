#!/bin/bash

for BS2 in 32 16 64; do
for S2 in 45 90 60 30; do
for LR2 in 0.0001 0.00008 0.0002; do
for NTUPLES in 1000 2000; do
for MARGIN in 0.12 0.05 0.1 0.15 0.18 0.20 0.22 0.25; do
for ARCH in 'Net' 'SmallNet' 'SmallDeepNet'; do
for DATA in 'blobs' 'edges'; do
    DATA_PATH=~/SLOW/Blobs/${DATA}-dataset.npz
    EXP_PATH=experiments/${DATA}

    python main.py ${DATA_PATH} \
                -d ${EXP_PATH} \
                -a ${ARCH}     \
                -m ${MARGIN}   \
               -l2 ${LR2}      \
               -e2 90          \
               -s2 ${S2}       \
                -n ${NTUPLES}  \
               -b2 ${BS2}
done
done
done
done
done
done
done

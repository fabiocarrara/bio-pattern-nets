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


# 'data', type=str, help='EDGES dataset file (npz)')

# '-z', '--seed', type=int, default=42, help='Random seed for reproducibility')
# '-a', '--modelArch', type=str, default='Net', choices=['Net', 'SmallNet', 'SmallDeepNet'], help='Model architecture [Net | SmallNet | SmallDeepNet] (default: Net)')
# '-st', '--skipTest', default=False, action='store_true', help='Skip the evaluation on the test set')

# STEP 1 PARAMS
# '-b', '--batchSize', type=int, default=32, help='Batch size for training (default: 32)')
# '-e', '--nEpochs', type=int, default=45, help='Number of epochs to train (default: 45)')
# '-l', '--learningRate', type=float, default=0.001, help='Initial learning rate (default: 0.001)')
# '-s', '--stepRateEpochs', type=int, default=15, help='Reduce LR after this number of epochs (default: 15)')
# '-i', '--logEvery', type=int, default=20, help='Number of iterations between two consecutive average loss logging (default: 20)')

# STEP 2 PARAMS
# '-m', '--margin', type=float, default=0.12, help='Margin value for ordinal loss (default: 0.12)')                                                                           
# '-n', '--nTuples', type=int, default=1000, help='How many ordered tuples to generate per epoch (default: 1000)')
# '-b2', '--batchSize2', type=int, default=32, help='Batch size for training [STEP 2] (default: 32)')
# '-e2', '--nEpochs2', type=int, default=30, help='Number of epochs to train [STEP 2] (default: 30)')
# '-l2', '--learningRate2', type=float, default=0.001, help='Initial learning rate [STEP 2] (default: 0.001)')
# '-s2', '--stepRateEpochs2', type=int, default=10, help='Reduce LR after this number of epochs [STEP 2] (default: 10)')
# '-i2', '--logEvery2', type=int, default=4, help='Number of iterations between two consecutive average loss logging [STEP 2] (default: 4)')

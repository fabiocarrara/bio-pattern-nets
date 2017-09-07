import argparse
import torch
import numpy as np

from common import init_workdir, log_experiment, already_done
from data import load_data
from train_step1 import step1
from train_step2 import step2
from eval import evaluate


def main():
    parser = argparse.ArgumentParser(description='Ordinal Regression for EDGES regression')
    parser.add_argument('data', type=str, help='BLOBS or EDGES dataset file (npz)')
    
    parser.add_argument('-d', '--workDir', type=str, default='', help='Base dir for all the generated files')
    parser.add_argument('-z', '--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('-a', '--modelArch', type=str, default='Net', choices=['Net', 'SmallNet', 'SmallDeepNet'], help='Model architecture [Net | SmallNet | SmallDeepNet] (default: Net)')
    parser.add_argument('-st', '--skipTest', default=False, action='store_true', help='Skip the evaluation on the test set')

    # STEP 1 PARAMS
    parser.add_argument('-b', '--batchSize', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('-e', '--nEpochs', type=int, default=45, help='Number of epochs to train (default: 45)')
    parser.add_argument('-l', '--learningRate', type=float, default=0.001, help='Initial learning rate (default: 0.001)')
    parser.add_argument('-s', '--stepRateEpochs', type=int, default=15, help='Reduce LR after this number of epochs (default: 15)')
    parser.add_argument('-i', '--logEvery', type=int, default=20, help='Number of iterations between two consecutive average loss logging (default: 20)')

    # STEP 2 PARAMS
    parser.add_argument('-m', '--margin', type=float, default=0.2, help='Margin value for ordinal loss (default: 0.2)')
    parser.add_argument('-n', '--nTuples', type=int, default=1000, help='How many ordered tuples to generate per '
                                                                         'epoch (default: 1000)')
    parser.add_argument('-b2', '--batchSize2', type=int, default=32, help='Batch size for training [STEP 2] (default: 32)')
    parser.add_argument('-e2', '--nEpochs2', type=int, default=90, help='Number of epochs to train [STEP 2] (default: 90)')
    parser.add_argument('-l2', '--learningRate2', type=float, default=0.0001, help='Initial learning rate [STEP 2] (default: 0.0001)')
    parser.add_argument('-s2', '--stepRateEpochs2', type=int, default=45, help='Reduce LR after this number of epochs [STEP 2] (default: 45)')
    parser.add_argument('-i2', '--logEvery2', type=int, default=4, help='Number of iterations between two consecutive average loss logging [STEP 2] (default: 4)')
    args = parser.parse_args()
    
    if already_done(args):
        print 'SKIPPING'
        return
    
    init_workdir(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    data = load_data(args)
    model, stats1 = step1(data, args)
    model, stats2 = step2(model, data, args)
    evaluate(model, data, args)
    
    log_experiment(args, stats1, stats2)
    

if __name__ == '__main__':
    main()

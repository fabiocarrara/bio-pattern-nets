#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import sys
import argparse
from random import shuffle
from itertools import chain, izip
from collections import Counter
from sklearn.cross_validation import train_test_split

def write_list_file(X, Y, classes, fname):
    with open(fname, 'wb') as f:
        for url, label in izip(X,Y):
            f.write('{} {}\n'.format(url, classes[label]))
        

def main():
    parser = argparse.ArgumentParser(description='Split the edges dataset.')
    parser.add_argument('edges_dir', type=str, help='Path to edges images')
    parser.add_argument('-b', '--balanced', action='store_true', help='Whether to produce a balanced train/val/test split or not')
    parser.add_argument('-v', '--val-size', type=float, default=0.10, help='Percentage of validation set')
    parser.add_argument('-t', '--test-size', type=float, default=0.10, help='Percentage of test set')
    parser.add_argument('-s', '--random-seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    classes = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    dset = [ (f, f[0]) for f in os.listdir(args.edges_dir) if f[0] in classes.keys() ]
    X, Y = zip(*dset)
    
    # print actual db stats
    print Counter(Y)
    write_list_file(X, Y, classes, 'all.txt')

    if args.balanced:
        
        # class => list of samples
        X_classes = { c: [i for i in X if i.startswith(c)] for c in classes.keys() }
        # number of samples in the minority class
        minClass = min([len(Xc) for Xc in X_classes.values()])
        
        Xbal_classes = {}
        # make it balanced taking 'minClass' random samples
        for c, Xc in X_classes.iteritems():
            shuffle(Xc)
            Xbal_classes[c] = Xc[0:minClass]
        
        Xb = list( chain(*Xbal_classes.values()) )
        Yb = list( chain(*[ [c,] * minClass for c in Xbal_classes.keys() ]) )
        
        X, Y = Xb, Yb
    
    trX, restX, trY, restY = train_test_split(X, Y,
                                test_size=(args.val_size + args.test_size),
                                random_state=args.random_seed,
                                stratify=Y)

    valX, testX, valY, testY = train_test_split(restX, restY,
                                test_size=args.test_size / (args.val_size + args.test_size),
                                random_state=args.random_seed,
                                stratify=restY)
    
    print 'Train set stats:'
    print Counter(trY)
    print 'Validation set stats:'
    print Counter(valY)
    print 'Test set stats:'
    print Counter(testY)
    
    write_list_file(trX, trY, classes, 'train.txt')
    write_list_file(valX, valY, classes, 'val.txt')
    write_list_file(testX, testY, classes, 'test.txt')

if __name__ == '__main__':
    sys.exit(main())
    




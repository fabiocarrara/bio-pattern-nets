#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os

os.environ['GLOG_minloglevel'] = '2'

import sys
import json
import caffe
import codecs
import shutil
import tarfile
import argparse
import tempfile
import numpy as np
from tqdm import tqdm
from bhtsne import tsne
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def bproto_to_numpy(fname):
    with open(fname, 'rb') as fhandler:
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = fhandler.read()
        blob.ParseFromString(data)
        return np.array( caffe.io.blobproto_to_array(blob) )


def main():
    parser = argparse.ArgumentParser(description='Visualize the edge dataset with TSNE.')
    parser.add_argument('model_archive', type=str, help='Path to the DIGITS\'s tar.gz model archive')
    parser.add_argument('image_folder', type=str, help='Path to images to be visualized')
    parser.add_argument('-l', '--layer', type=str, default='res3_sum', help='Which layer output to use as feature of an image')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size for feature extraction')
    
    args = parser.parse_args()
    
    tmpdir = tempfile.mkdtemp()
    
    urls_and_labels = [(os.path.join(args.image_folder, fname), fname[0]) for fname in os.listdir(args.image_folder) if fname.endswith('png')]
    
    #urls_and_labels = urls_and_labels[0:4*args.batch_size]
    
    urls, labels = zip(*urls_and_labels)
    classes = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    classes_idx = ['A','B','C','D','E']
    numeric_labels = np.array([classes[i] for i in labels])
    
    features_fname = args.model_archive.split(os.extsep, 1)[0] + "_" + args.layer + ".npy"
    predictions_fname = args.model_archive.split(os.extsep, 1)[0] + "_predictions.npy"
    
    if os.path.exists(features_fname) and os.path.exists(predictions_fname):
        features = np.load(features_fname, mmap_mode='r')
        predictions = np.load(predictions_fname)
    else: # extract features!
        model_archive = tarfile.open(args.model_archive, 'r:gz')
        model_archive.extractall(tmpdir)
        model_archive.close()
            
        info = json.load(open(os.path.join(tmpdir, 'info.json'), 'rb'))
        prototxt = os.path.join(tmpdir, 'deploy.prototxt')
        weights = os.path.join(tmpdir, info['snapshot file'])
        mean = bproto_to_numpy(os.path.join(tmpdir, 'mean.binaryproto'))[0]
        mean_pixel = mean.mean(1).mean(1)
        
        # XXX trouble with caffe and unicode strings.. need to use str()
        net = caffe.Net(prototxt, caffe.TEST, weights=str(weights))
        
        net.blobs['data'].reshape(args.batch_size, *net.blobs['data'].data.shape[1:])
        
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_mean('data', mean_pixel)
        transformer.set_channel_swap('data', (2, 1, 0)) # RGB -> BGR
        transformer.set_raw_scale('data', 255) # [0,1] -> [0,255]
        transformer.set_transpose('data', (2, 0, 1)) # HWC -> CHW
    
        url_batches = [urls[i:i + args.batch_size] for i in xrange(0, len(urls), args.batch_size)]
        
        # fully conv feature is globally max pooled over the spatial dimensions and l2-normalized
        # TODO try RMAC!
        feature_size = net.blobs[args.layer].data[0].size
        
        features = np.empty((len(urls), feature_size), dtype=np.float32)
        predictions = np.empty((len(urls)), dtype=np.int32)
    
        print 'Feature shape:', net.blobs[args.layer].data.shape[1:]
        
        b = 0
        for batch in tqdm(url_batches):
            if len(batch) != net.blobs['data'].data.shape[0]:
                net.blobs['data'].reshape(len(batch), *net.blobs['data'].data.shape[1:])
            
            for i, url in enumerate(batch):
                img = caffe.io.load_image(url)
                side = net.blobs['data'].data.shape[-1]
                start = (info['image dimensions'][0] - side) // 2
                crop = img[start:start+side, start:start+side]
                net.blobs['data'].data[i, ...] = transformer.preprocess('data', crop)
            
            out = net.forward()
            # collect classifications
            predictions[b:b + len(batch), ...] = np.argmax(out.values()[0].squeeze(), axis=1)
            
            # collect features
            # out = net.blobs[args.layer].data.max(2).max(2) # get max global pool and l2-normalize
            out = net.blobs[args.layer].data
            flatten = out.reshape((out.shape[0], -1))
            
            features[b:b + len(batch), ...] = normalize(flatten)
            b += len(batch)
        
        del net
        del url_batches
        del transformer
        
        np.save(features_fname, features)
        np.save(predictions_fname, predictions)
        
    print predictions
        
    embed = tsne(features.astype(np.float64))
    
    # save all to JSON
    tojson = [ dict(
        x=embed[i,0],
        y=embed[i,1],
        url=urls[i],
        label=labels[i],
        nlabel=numeric_labels[i],
        prediction=int(predictions[i]) ) for i in xrange(0, len(urls)) ]
    
    json.dump(tojson, codecs.open('tsne.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
    
    del tojson
    
    # plot them by class!
    colors = ['r', 'b', 'g', 'y', 'm']
    for cindex, cname in enumerate(classes_idx):
        XY = embed[numeric_labels == cindex, :]
        X, Y = (XY[:, 0], XY[:, 1])
        plt.scatter(X,Y, label=cname, c=colors[cindex], edgecolors='none')
    
    plt.legend(numpoints=1)
    plt.show()
        
    shutil.rmtree(tmpdir)
    
if __name__ == '__main__':
    sys.exit(main())



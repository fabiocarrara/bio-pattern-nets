#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sys
import caffe
import argparse
from caffe import layers as L, params as P

def residual_block(n, bottom, first_stride=2, num_output=64, avg_pool_reduction=False):
    
    residual_block.counter += 1
    
    conv_defaults = dict(
        kernel_size=3,
        pad=1,
        weight_filler=dict(type='msra'),
        bias_term=False
    )
    
    bn_defaults = dict(param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
    
    prefix = 'res{}_'.format(residual_block.counter)
    n[prefix + 'conv1'] = L.Convolution(bottom, stride=first_stride, num_output=num_output, **conv_defaults)
    n[prefix + 'bn1'] = L.BatchNorm(n[prefix + 'conv1'], **bn_defaults)
    n[prefix + 'relu'] = L.ReLU(n[prefix + 'bn1'], in_place=True)
    n[prefix + 'conv2'] = L.Convolution(n[prefix + 'relu'], stride=1, num_output=num_output, **conv_defaults)
    n[prefix + 'bn2'] = L.BatchNorm(n[prefix + 'conv2'], **bn_defaults)
    
    # dimensionality changes
    if first_stride != 1 and avg_pool_reduction:
        # fix width and height
        n[prefix + 'x_reduced'] = L.Pooling(bottom, stride=first_stride, pad=1, kernel_size=3, pool=P.Pooling.AVE)
        bottom = n[prefix + 'x_reduced']
        first_stride = 1 # change it for evetual channel fix
        
    # fix channels if there's a change
    if residual_block.last_num_output != num_output:
        n[prefix + 'x_proj'] = L.Convolution(bottom, stride=first_stride, kernel_size=1, num_output=num_output, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        bottom = n[prefix + 'x_proj'] 
            
    n[prefix + 'sum'] = L.Eltwise(bottom, n[prefix + 'bn2'], operation=P.Eltwise.SUM)
    n[prefix + 'out'] = L.ReLU(n[prefix + 'sum'], in_place=True)
    
    residual_block.last_num_output = num_output
    
    return n[prefix + 'out']

residual_block.counter = 0
residual_block.last_num_output = 16

def main():
    parser = argparse.ArgumentParser(description='Build a model for Edge Classification')
    parser.add_argument('output_prototxt', type=str, help='Output prototxt file')
    parser.add_argument('-n', '--num-classes', type=int, default=5, help='Number of output classes (default 5)')
    
    args = parser.parse_args()
    
    n = caffe.NetSpec()
    n['conv1'] = L.Convolution(bottom='data', kernel_size=3, stride=1, pad=1, num_output=16, weight_filler=dict(type='msra'), bias_term=False)
    n['bn1'] = L.BatchNorm(n['conv1'], param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
    n['relu1'] = L.ReLU(n['bn1'], in_place=True)
    
    top = residual_block(n, n['relu1'], num_output=64, first_stride=1)
    top = residual_block(n, top, num_output=128)
    top = residual_block(n, top, num_output=256)
    
    n['conv_final'] = L.Convolution(top, kernel_size=3, stride=1, pad=1, num_output=args.num_classes, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    n['relu_final'] = L.ReLU(n['conv_final'], in_place=True)
    
    n['global_pool'] = L.Pooling(n['relu_final'], pool=P.Pooling.AVE, global_pooling=True)
    n['loss'] = L.SoftmaxWithLoss(bottom=['global_pool', 'label'], include=[dict(phase=caffe.TRAIN), dict(phase=caffe.TEST, stage='val')], exclude=dict(phase=caffe.TEST, stage='deploy') )
    n['accuracy'] = L.Accuracy(bottom=['global_pool', 'label'], include=dict(phase=caffe.TEST, stage='val'))
    n['prob'] = L.Softmax(bottom='global_pool', include=dict(phase=caffe.TEST, stage='deploy'))
    
    net = n.to_proto()
    net.name = 'EdgeNet-Res'
    
    with open(args.output_prototxt, 'wb') as f:
        f.write(str(net))
    
if __name__ == '__main__':
    sys.exit(main())



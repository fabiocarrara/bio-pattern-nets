# Blobs regression

These files contain the model for regression of the Blob dataset.

## Steps to train the model
1. Generate train, val and test LMDB databases from the list files contained in `dataset/`. This can be done using [DIGITS](https://developer.nvidia.com/digits).
2. Use DIGITS or the `caffe` binary to train the model. 
You can train the model with **Euclidean Loss**:
```sh
caffe train -solver solver-Euc.prototxt
```
or the one with **Relaxed Euclidean Loss**:
```sh
caffe train -solver solver.prototxt
```



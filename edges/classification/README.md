# Classification of Edges dataset

These files contain the code to reproduce the training of the classifier on the Edges dataset.

`1_split_dataset.py`: Splits the dataset in train/val/test subsets.
```sh
$ python 1_split_dataset.py -h
usage: 1_split_dataset.py [-h] [-b] [-v VAL_SIZE] [-t TEST_SIZE]
                          [-s RANDOM_SEED]
                          edges_dir

Split the edges dataset.

positional arguments:
  edges_dir             Path to edges images

optional arguments:
  -h, --help            show this help message and exit
  -b, --balanced        Whether to produce a balanced train/val/test split or
                        not
  -v VAL_SIZE, --val-size VAL_SIZE
                        Percentage of validation set
  -t TEST_SIZE, --test-size TEST_SIZE
                        Percentage of test set
  -s RANDOM_SEED, --random-seed RANDOM_SEED
                        Random seed

```
`2_build_net.py`: Generates a prototxt file containing the classifier model.
```sh
$ python 2_build_net.py -h
usage: 2_build_net.py [-h] [-n NUM_CLASSES] output_prototxt

Build a model for Edge Classification

positional arguments:
  output_prototxt       Output prototxt file

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_CLASSES, --num-classes NUM_CLASSES
                        Number of output classes (default 5)
```

The generated model has been trained using [DIGITS](https://developer.nvidia.com/digits).
The pretrained model together with all the training hyper-parameters can be found here: [trained_model.tar.gz](http://pc-carrara.isti.cnr.it/Blobs/trained_model.tar.gz) (4.3 MB)

`3_classify_and_tsne.py`: Classifies a list of files and produces their TSNE visualization given a trained model archive.
```sh
$ python 3_classify_and_tsne.py  -h
usage: 3_classify_and_tsne.py [-h] [-l LAYER] [-b BATCH_SIZE]
                              model_archive image_folder

Visualize the edge dataset with TSNE.

positional arguments:
  model_archive         Path to the DIGITS's tar.gz model archive
  image_folder          Path to images to be visualized

optional arguments:
  -h, --help            show this help message and exit
  -l LAYER, --layer LAYER
                        Which layer output to use as feature of an image
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for feature extraction
```

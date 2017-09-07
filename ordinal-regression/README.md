# BioPattern Scoring with Ordinal Regression Loss

 Code for training and using Convolutional Neural Networks for scoring biological patterns images.

## Requirements and Setup

 - Python 2.7
 - PyTorch
 
 You can setup the environment with the following commands:
 
 1. Clone this repo and change directory:
 ```bash
 git clone git@github.com:fabiocarrara/bio-pattern-nets.git
 cd bio-pattern-nets/ordinal-regression
 ```
 2. Setup a Python Virtual Environment and activate it:
 ```bash
 virtualenv --system-site-packages venv
 source venv/bin/activate
 ```
 3. Install python package requirements:
 ```bash
 pip install -r requirements.txt
 ```
 4. Install PyTorch following the instructions [here](http://pytorch.org/) (choose 'pip' as package manager and '2.7' as Python).

## Usage

 To score images you can use this command:
 ```bash
 python score.py <trained_model.th> <image_files_or_folder> --arch <architecture> --batchSize <bs> [--gpu]
 ```
 
**Make sure the `arch` you specify corresponds to the one of the model you load.**
 
 Some pretrained model you can download and use:
 
|Model|Train Data|Architecture|Test Rank Accuracy|Test Set Scoring|
|-----|:---:|------------|---:|:---:|
[blobs_2017-09-01_115027.th](http://pc-carrara.isti.cnr.it/bio-patterns/models/blobs_2017-09-01_115027.th)|BLOBS|Net|99.54 %|[Results](http://pc-carrara.isti.cnr.it/bio-patterns/results/results_blobs_2017-09-01_115027.html)
[blobs_smalldeep_2017-09-02_220737.th](http://pc-carrara.isti.cnr.it/bio-patterns/models/blobs_smalldeep_2017-09-02_220737.th)|BLOBS|SmallDeepNet|99.15 %|[Results](http://pc-carrara.isti.cnr.it/bio-patterns/results/results_blobs_smalldeep_2017-09-02_220737.html)
[blobs_small_2017-09-02_024658.th](http://pc-carrara.isti.cnr.it/bio-patterns/models/blobs_small_2017-09-02_024658.th)|BLOBS|SmallNet|95.11 %|[Results](http://pc-carrara.isti.cnr.it/bio-patterns/results/results_blobs_small_2017-09-02_024658.html)
[edges_2017-08-28_225235.th](http://pc-carrara.isti.cnr.it/bio-patterns/models/edges_2017-08-28_225235.th)|EDGES|Net|96.65 %|[Results](http://pc-carrara.isti.cnr.it/bio-patterns/results/results_edges_2017-08-28_225235.html)
[edges_smalldeep_2017-09-05_050355.th](http://pc-carrara.isti.cnr.it/bio-patterns/models/edges_smalldeep_2017-09-05_050355.th)|EDGES|SmallDeepNet|95.64 %|[Results](http://pc-carrara.isti.cnr.it/bio-patterns/results/results_edges_smalldeep_2017-09-05_050355.html)
[edges_small_2017-09-05_042616.th](http://pc-carrara.isti.cnr.it/bio-patterns/models/edges_small_2017-09-05_042616.th)|EDGES|SmallNet|97.30 %|[Results](http://pc-carrara.isti.cnr.it/bio-patterns/results/results_edges_small_2017-09-05_042616.html)

### Usage examples
 - Score a BLOB image using a SmallNet, print the results to stdout:
 ```bash
 python score.py blobs_small_2017-09-02_024658.th --arch SmallNet my_blob_image.png 
 ```

 - Score a folder full of EDGES images:
 ```bash
 python score.py edges_2017-08-28_225235.th path/to/my_folder_full_of_edges_images/ 
 ```

 - Score a folder full of EDGES images using GPU acceleration (if you have CUDA installed), save the scores on file:
 ```bash
 python score.py edges_2017-08-28_225235.th path/to/my_folder_full_of_edges_images/ --gpu --batchSize 256 > scores.tsv
 ```

## Training

 A model is trained in two steps. First, the model is pretrained as a N-way classifier (no scores are obtained).
 In the second step, we substitute the classification layers with a scoring layer, giving scores in the [0, 1] interval.
 The whole model is than fine-tuned using a N-way ordinal-regression loss.

 In particular, we give to the network N images, one for each class (eg. A, B, C, D, E), as input, and we ask the network
 to score the images in order to be correctly ranked (scores from A to E have to be sorted correctly). This training step
 created a regressor that output more uniform scores in the [0, 1] interval. 

 To train the model you need the BLOBS and EDGES datasets in `.npz` packaged format. You can download them from here:
 
 - [blobs-dataset.npz](http://pc-carrara.isti.cnr.it/bio-patterns/blobs-dataset.npz) (41.7MB)
 - [edges-dataset.npz](http://pc-carrara.isti.cnr.it/bio-patterns/edges-dataset.npz) (64.1MB)

 You can train the full architecture model (i.e: _Net_ model) on the EDGES dataset like this:
 
 ```bash
 python main.py path/to/edges-dataset.npz --workDir edges_training/
 ```
 At the end of the training procedure, you can find the model snapshot in `<workDir>/snapshots_2/model_best_loss.th`.

 Check the manual of the `main.py` script for a description of all the training parameters.
 ```bash
 python main.py -h
 ```


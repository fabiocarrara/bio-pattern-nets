import os

import math
import time

import torch
import torch.nn.functional as F
import torch.optim

from tqdm import tqdm, trange
from torch.autograd import Variable as V

from data import to_ordinal_data
from common import save_model, load_model, exp_lr_scheduler


def train(model, optimizer, epoch, loader, logfile, args):

    model.train(True)  # Set model to training mode
    optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=args.learningRate2, lr_decay_epoch=args.stepRateEpochs2)

    running_loss = 0.0
    display_loss = 0.0
    
    dataset_size = len(loader.dataset)
    tot_iterations = math.ceil(dataset_size / float(args.batchSize2))
    tot_iterations = int(tot_iterations)
    # Iterate over data.
    for iteration, data in tqdm(enumerate(loader), desc='Iterations',
                                total=tot_iterations):
        # get the inputs
        inputs = data
        # wrap them in Variable
        inputs = [V(i.cuda()) for i in inputs]
                
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = [model(i) for i in inputs]
        diffs = torch.stack([(outputs[i] - outputs[i+1]) for i in xrange(len(outputs)-1)])
        loss = F.relu(args.margin + diffs).sum()

        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()
        
        # statistics
        running_loss += loss.data[0]
        display_loss += loss.data[0]
        
        if iteration % args.logEvery2 == 0:
            avg_loss = (display_loss / args.logEvery2) if iteration else display_loss
            logfile.write('T {} {} {}\n'.format(epoch, iteration, avg_loss))
            logfile.flush()
            display_loss = 0
                
    epoch_average_loss = running_loss / dataset_size

    #tqdm.write('TRAIN Loss: {:.4f}'.format(phase.upper(), epoch_loss)
    
    return epoch_average_loss

    
def evaluate(model, loader, args):
    model.train(False)  # Set model to evaluate mode

    running_loss = 0.0
    display_loss = 0.0
    correct = 0
    
    dataset_size = len(loader.dataset)
    tot_iterations = math.ceil(dataset_size / float(args.batchSize2))
    tot_iterations = int(tot_iterations)
    # Iterate over data.
    for iteration, data in tqdm(enumerate(loader), desc='Iterations',
                                total=tot_iterations):
        # get the inputs
        inputs = data
        # wrap them in Variable
        inputs = [V(i.cuda(), volatile=True) for i in inputs]
        # forward
        outputs = torch.stack([model(i) for i in inputs]).squeeze()
        # compute diffs of adjacent rankings
        diffs = outputs[:-1] - outputs[1:]
        loss = F.relu(args.margin + diffs).sum()
        # statistics
        running_loss += loss.data[0]
        # count the number of correctly sorted tuples (cols having all diffs < 0)
        correct += ((diffs < 0).float().sum(0) == diffs.size(0)).float().sum().cpu()
        # TODO make a Counter with all different intermediate rankings countings
    
    correct = correct.data[0]
    epoch_loss = running_loss / dataset_size
    epoch_accuracy = 100. * correct / dataset_size
    
    return epoch_loss, epoch_accuracy, correct


def step2(model, data, args):
    print '### STEP 2: Train for ordinal regression task'

    pretrained_snapshot_fname = 'model_best_loss.th'

    train_loader, val_loader, test_loader = to_ordinal_data(data, args)

    n_samples_train = len(train_loader.dataset)
    n_samples_val = len(val_loader.dataset)
    n_samples_test = len(test_loader.dataset)

    best_val_acc = None
    test_acc = None
    
    model.to_ordinal()
    saved_model = load_model(model, pretrained_snapshot_fname, args, subdir='snapshots_2')
    if saved_model is not None:
        print 'Loading pretrained model:', pretrained_snapshot_fname
        model = saved_model
        model.cuda()
    else:
        logfile = open(os.path.join(args.workDir, 'log_2.txt'), 'wb')
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), args.learningRate2)

        since = time.time()
        for epoch in trange(args.nEpochs2 + 1, desc='Epochs'):
            avg_loss = train(model, optimizer, epoch, train_loader, logfile, args)
            val_loss, val_acc, n_correct = evaluate(model, val_loader, args)
            train_loader.dataset.sample_tuples()
            val_loader.dataset.sample_tuples()

            if best_val_acc is None or best_val_acc < val_acc:
                best_val_acc = val_acc
                tqdm.write('Snapshotting best model: ' + pretrained_snapshot_fname)
                save_model(model, pretrained_snapshot_fname, args, subdir='snapshots_2')

            logline = 'Epoch {:3d}/{}] train_avg_loss = {:.4f}, val_avg_loss = {:.4f}, val_accuracy = {}/{} ({:.2f}%, Best: {:.2f}%)'
            tqdm.write(logline.format(epoch, args.nEpochs2, avg_loss, val_loss, n_correct, n_samples_val, val_acc, best_val_acc))

        time_elapsed = time.time() - since
        print 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
        model = load_model(model, pretrained_snapshot_fname, args, subdir='snapshots_2')
        model.cuda()

    # RANK TESTING ------------
    if not args.skipTest:
        test_loss, test_acc, n_correct = evaluate(model, test_loader, args)
        logline = 'TEST] test_avg_loss = {:.4f}, test_accuracy = {}/{} ({:.2f}%)'
        print logline.format(test_loss, n_correct, n_samples_test, test_acc)

    return model, {'Best Val Rank Accuracy': best_val_acc, 'Test Rank Accuracy': test_acc}


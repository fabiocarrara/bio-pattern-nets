import os

import math
import time

import torch
import torch.nn.functional as F
import torch.optim

from tqdm import tqdm, trange
from torch.autograd import Variable as V

from model import Net, SmallNet, SmallDeepNet
from common import save_model, load_model, exp_lr_scheduler


def train(model, optimizer, epoch, loader, logfile, args):

    model.train(True)  # Set model to training mode
    optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=args.learningRate, lr_decay_epoch=args.stepRateEpochs)

    running_loss = 0.0
    display_loss = 0.0
    
    dataset_size = len(loader.dataset)
    tot_iterations = math.ceil(dataset_size / float(args.batchSize))
    tot_iterations = int(tot_iterations)
    # Iterate over data.
    for iteration, data in tqdm(enumerate(loader), desc='Iterations',
                                total=tot_iterations):
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        inputs, labels = V(inputs.cuda()), V(labels.cuda())
                
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)

        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()
        
        # statistics
        running_loss += loss.data[0]
        display_loss += loss.data[0]
        
        if iteration % args.logEvery == 0:
            avg_loss = (display_loss / args.logEvery) if iteration else display_loss
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
    tot_iterations = math.ceil(dataset_size / float(args.batchSize))
    tot_iterations = int(tot_iterations)
    # Iterate over data.
    for iteration, data in tqdm(enumerate(loader), desc='Iterations',
                                total=tot_iterations):
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        inputs = V(inputs.cuda(), volatile=True)
        labels = V(labels.cuda(), volatile=True)
        
        # forward
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        predictions = torch.max(outputs.data, 1, True)[1]
        correct += predictions.eq(labels.data.view_as(predictions)).sum()

        # statistics
        running_loss += loss.data[0]
        
    epoch_loss = running_loss / dataset_size
    epoch_accuracy = 100. * correct / dataset_size
            
    return epoch_loss, epoch_accuracy, correct
    

def step1(data, args):
    print '### STEP 1: Train for classification task'
    
    pretrained_snapshot_fname = 'model_best_accuracy.th'
    
    train_loader, val_loader, test_loader = data
    
    n_samples_train = len(train_loader.dataset)
    n_samples_val = len(val_loader.dataset)
    n_samples_test = len(test_loader.dataset)

    num_classes = len(set(val_loader.dataset.target_tensor))
    
    model = eval(args.modelArch)(num_classes=num_classes)
    
    best_val_acc = None
    test_acc = None

    # try to load pretrained model if step 1 has already been executed
    saved_model = load_model(model, pretrained_snapshot_fname, args)
    if saved_model is not None:
        print 'Loading pretrained model:', pretrained_snapshot_fname
        model = saved_model
        model.cuda()
    else: 
        # else train a new model
        print 'Training a new model ...'    
        logfile = open(os.path.join(args.workDir, 'log.txt'), 'wb')
        
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), args.learningRate)
        
        since = time.time()
        for epoch in trange(1, args.nEpochs + 1, desc='Epochs'):
            avg_loss = train(model, optimizer, epoch, train_loader, logfile, args)
            val_loss, val_acc, n_correct = evaluate(model, val_loader, args)
            
            if best_val_acc is None or best_val_acc < val_acc:
                best_val_acc = val_acc
                tqdm.write('Snapshotting best model: ' + pretrained_snapshot_fname)
                save_model(model, pretrained_snapshot_fname, args)
            
            logline = 'Epoch {:3d}/{}] train_avg_loss = {:.4f}, val_avg_loss = {:.4f}, val_accuracy = {}/{} ({:.2f}%, Best: {:.2f}%)'
            tqdm.write(logline.format(epoch, args.nEpochs, avg_loss, val_loss, n_correct, n_samples_val, val_acc, best_val_acc))
            
        time_elapsed = time.time() - since
        print 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
        model = load_model(model, pretrained_snapshot_fname, args)

    # TESTING -----------------
    if not args.skipTest:
        test_loss, test_acc, n_correct = evaluate(model, test_loader, args)
        logline = 'TEST] test_avg_loss = {:.4f}, test_accuracy = {}/{} ({:.2f}%)'
        print logline.format(test_loss, n_correct, n_samples_test, test_acc)
    
    return model, {'BestValAccuracy': best_val_acc, 'TestAccuracy': test_acc}

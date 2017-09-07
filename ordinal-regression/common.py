import os
import datetime
import torch

import numpy as np
import pandas as pd

results_fname = 'experiments.csv'


def init_workdir(args):        
    if not os.path.exists(args.workDir):
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        if args.workDir:
            args.workDir += '_' + now
        else:
            args.workDir = now
            
        os.mkdir(args.workDir)
        
    params_fname = os.path.join(args.workDir, 'params.txt')
    with open(params_fname, 'wb') as f:
        for key, value in vars(args).iteritems():
            f.write('{}: {}\n'.format(key, value))


def save_model(model, snapshot_fname, args, subdir='snapshots'):
    snapshot_dir = os.path.join(args.workDir, subdir)
    if not os.path.exists(snapshot_dir):
        os.mkdir(snapshot_dir)
    snapshot_fname = os.path.join(snapshot_dir, snapshot_fname)
    torch.save(model.state_dict(), snapshot_fname)


def load_model(model, snapshot_fname, args, subdir='snapshots'):
    snapshot_dir = os.path.join(args.workDir, subdir)
    snapshot_fname = os.path.join(snapshot_dir, snapshot_fname)
    
    if not os.path.exists(snapshot_fname):
        print 'Non-existant snapshot:', snapshot_fname
        return None

    model.load_state_dict(torch.load(snapshot_fname))
    return model
    

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print 'LR is set to {}'.format(lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
    

def log_experiment(args, *stats):
    global results_fname
    
    if len(stats) == 0:
        return

    data = vars(args)
    for s in stats:
        data.update(s)
    
    # XXX: append mode should be more efficient,
    # but there's no check on columns... sticking with this
    if os.path.exists(results_fname):
        dtypes = {k: type(v) for k, v in data.iteritems()}
        results = pd.read_csv(results_fname, dtype=dtypes)
        results = results.append([data], ignore_index=True)
    else:
        results = pd.DataFrame([data])
    
    results.to_csv(results_fname, index=False)
    

def already_done(args):
    global results_fname
    
    if os.path.exists(results_fname):
        data = vars(args).copy()
        del data['workDir']
        current_exp_row = np.array(data.values(), dtype=np.object)
        
        dtypes = {k: type(v) for k, v in data.iteritems()}
        experiments = pd.read_csv(results_fname, dtype=dtypes)
        
        if (experiments[data.keys()].values == current_exp_row).all(1).any():
            return True
            
    return False


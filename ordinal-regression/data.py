import torch
import numpy as np

from collections import Counter
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset, Dataset


def load_data(args):
    # load whole dataset in memory
    print 'Loading the dataset in memory:', args.data
    dataset = np.load(args.data)
    X = dataset['X'].astype(np.float32) # convert uint8 to float32
    Y = dataset['Y'] - 1 # 1-based to 0-based label
    
    # transpose dataset from NHWC to to NCHW
    X = np.transpose(X, (0, 3, 1, 2))
    
    # split dataset in train, val, and test
    train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.33, random_state=args.seed, stratify=Y)
    val_X, test_X, val_Y, test_Y = train_test_split(val_X, val_Y, test_size=0.50, random_state=args.seed, stratify=val_Y)
    
    # convert to torch tensor
    train_X = torch.from_numpy(train_X)
    train_Y = torch.from_numpy(train_Y)
    val_X = torch.from_numpy(val_X)
    val_Y = torch.from_numpy(val_Y)
    test_X = torch.from_numpy(test_X)
    test_Y = torch.from_numpy(test_Y)
    
    print 'Train Set:', train_X.size(), Counter(train_Y)
    print 'Val Set:', val_X.size(), Counter(val_Y)
    print 'Test Set:', test_X.size(), Counter(test_Y)

    train_data = TensorDataset(train_X, train_Y)
    val_data = TensorDataset(val_X, val_Y)
    test_data = TensorDataset(test_X, test_Y)
    
    train_loader = DataLoader(train_data, batch_size=args.batchSize, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=args.batchSize, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=args.batchSize, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


class OrderedTupleDataset (Dataset):
    """
    Dataset to create a N-ordered-tuple dataset from a TensorDataset.
    Only TensorDataset is supported as underlying dataset for now.
    """

    def __init__(self, dataset, limit=1000):
        self.limit = limit
        self.data_tensor = dataset.data_tensor
        
        labels = dataset.target_tensor.numpy().squeeze()
        unique_labels = np.unique(labels)
        
        # build a label => samples indices (labels are ordered by unique)
        self.label_idx = {label: (labels == label).nonzero()[0] for label in unique_labels}
        self.sample_tuples()
    
    def sample_tuples(self):
        # get 5 columns of random indices, each beloning to one class
        random_samples_per_label = [np.random.choice(samples, self.limit) for samples in self.label_idx.values()]
        # merge the 5 columns together to build the tuples dataset
        self.tuples = np.vstack(random_samples_per_label).T
        
    def __getitem__(self, index):
        return [self.data_tensor[i] for i in self.tuples[index]]
    
    def __len__(self):
        return self.limit


def to_ordinal_data(data, args):
    train_data = OrderedTupleDataset(data[0].dataset, limit=args.nTuples)
    val_data = OrderedTupleDataset(data[1].dataset, limit=10**4)
    test_data = OrderedTupleDataset(data[2].dataset, limit=10**4)
    
    train_loader = DataLoader(train_data, batch_size=args.batchSize, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=args.batchSize, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=args.batchSize, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
    

if __name__ == '__main__':
    X = torch.randn(10, 3, 100, 100)
    Y = torch.LongTensor([3, 2, 1, 0, 2, 3, 1, 0, 2, 1])
    dset = TensorDataset(X,Y)
    dset = OrderedTupleDataset(dset, limit=10)
    loader = DataLoader(dset, batch_size=2)
    
    for a, b, c, d in loader:
        print a.shape, b.shape, c.shape, d.shape
    
    




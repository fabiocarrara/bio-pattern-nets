import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()        
        self.is_ordinal = False
        self.features = None
        self.classifier = None
        self.out_dim = 256
    
    def initialize_weights(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_normal(m.weight, gain=init.calculate_gain('relu'))
                init.constant(m.bias, 0.1)
                
        self.apply(weights_init)

    def forward(self, x):
        # print x.size()
        x = self.features(x)
        # print x.size()
        x = F.avg_pool2d(x, kernel_size=x.size()[2:]) # Global Average Pooling
        # print x.size()
        x = x.view(x.size()[0], -1)
        # print x.size()
        x = self.classifier(x)
        # print x.size()
        return x
        
    def to_ordinal(self):
        if not self.is_ordinal:
            m = nn.Linear(self.out_dim, 1)
            init.xavier_normal(m.weight, gain=init.calculate_gain('sigmoid'))
            init.constant(m.bias, 0.)
            self.classifier = nn.Sequential(
                nn.Dropout(),
                m,
                nn.Sigmoid())
            self.is_ordinal = True


class Net(BaseNet):
    def __init__(self, num_classes=5):
        super(Net, self).__init__()
        
        dilation = 1
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, dilation=dilation, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, dilation=dilation, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, dilation=dilation, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, dilation=dilation, stride=2),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        
        self.initialize_weights()
        

class SmallNet(BaseNet):
    def __init__(self, num_classes=5):
        super(SmallNet, self).__init__()
        
        dilation=1
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, dilation=dilation, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, dilation=dilation, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, dilation=dilation, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, dilation=dilation, stride=2),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )
        
        self.initialize_weights()


class SmallDeepNet(BaseNet):
    def __init__(self, num_classes=5):
        super(SmallDeepNet, self).__init__()
        
        dilation=1
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, dilation=dilation, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, dilation=dilation, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=dilation, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=dilation, stride=2),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64, num_classes),
        )
        
        self.out_dim = 64        
        self.initialize_weights()
        
                
if __name__ == '__main__':
    inputs = Variable(torch.randn(10, 3, 100, 100))
    model = eval('SmallDeepNet')()
    print model
    outputs = model(inputs)
    

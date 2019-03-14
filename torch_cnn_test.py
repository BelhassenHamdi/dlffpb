from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from torch.jit import ScriptModule, script_method, trace
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import datetime
import ast
import logging

parser = argparse.ArgumentParser(description='Merged test to benchmark tensorflow regualr compiler\
                                            vs tensorflow xla compiler vs torch vs torch new compiler.')
parser.add_argument('--test', type=ast.literal_eval,default=True, help='would inform the script if this is just a test run or a full run')

args = parser.parse_args()

currentDT = str(datetime.datetime.now()).split('.')[0]

if args.test:
    logfile_path="logs/torch_test_logs.log"
    name_suffix='test'
else:
    logfile_path='logs/torch_run_'+currentDT+'.log'
    name_suffix=currentDT

logging.basicConfig(filename=logfile_path, level=logging.DEBUG)
batch_size = 16
size = (batch_size, 3, 256, 256)
test_epochs = [10, 15, 20, 150, 300]
epochs = np.ceil(np.logspace(1, 3, 30, endpoint=False)).astype('int')
mu = 0
sigma = 2
size2 = (batch_size, 64)
x_val = np.random.normal(mu, sigma, size) 
y_val = np.random.normal(mu, sigma, size2)
z_val = np.random.normal(mu, sigma, size2)

if args.test:
    epochs = test_epochs

class SimpleCNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(32 * 256 * 256, 64)
        
        #64 input features, 10 output features for our 10 defined classes
        # self.fc2 = torch.nn.Linear(64, 10)
    # @script_method
    def forward(self, a):
        x, y, z = a
        #Computes the activation of the first convolution
        #Size changes from (3, 256, 256) to (18, 256, 256)
        x = self.conv1(x)

        #Size changes from (18, 256, 256) to (18, 128, 128)
        # x = self.pool(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 128, 128) to (1, 16384)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 32 * 256 * 256)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 16384) to (1, 64)
        x = F.elu(self.fc1(x))
        x = x + y * z
        x = x.sum()
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        # x = self.fc2(x)
        return(x)

def run_torch():
    
    m = SimpleCNN()
    m.cuda()
    start = time. time()
    for _ in range(epoch):
        x_val_ten, y_val_ten, z_val_ten = (torch.tensor(i).float().cuda() for i in [x_val, y_val, z_val] )
        out = m.forward([x_val_ten, y_val_ten, z_val_ten])
    end = time. time()
    p_t = end - start
    return out, p_t

tt = []

file = open('test1_results.txt','w+')
for epoch in epochs:
    torchoutput, tp = run_torch()
    tt.append(tp)
    file.write('{} {}\n'.format(str(epoch),str(tp)))
    print('epoch {}, execution time {}'.format(epoch,tp))
    logging.info("epoch num: {}, execution time {}".format(epoch, tt))

plt.figure()
plt.xlabel('number of epochs')
plt.ylabel('execution time')
plt.title('Pytorch compiler')
plt.plot(epochs, tt, 'k-o', label='pytorch compiler')
# plt.yscale('log')
# plt.xscale('log')
plt.legend()
plt.savefig('results/exectimeTorch_'+name_suffix)
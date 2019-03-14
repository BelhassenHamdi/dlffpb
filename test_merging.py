from torch.autograd import Variable                 #neither sure about this one as well
import torch.nn.functional as F                     #still not really sure about this one
import torch
import torchvision                                      
import torchvision.transforms as transforms         #pytorch library which implements some image processing function
import numpy as np
import tensorflow as tf
from tensorflow.contrib.compiler import xla         #xla compiler from tensorflow
import logging                                      #logging information to disc to find it in the future
import time
import matplotlib
from importlib import reload                        # force reloading modules and libraries especially usefull when you are builing and testing your own
matplotlib.use('Agg')                               # make matplot don't use the UI cause we are on docker and can't access the UI
import matplotlib.pyplot as plt                     # graph and plots library
import gc                                           # garbage collection from python frees and deletes unused variables 
import resource                                     #checks memory usage 
import argparse                                     #set flags and parse arguments more friendly
import ast
import datetime

parser = argparse.ArgumentParser(description='Merged test to benchmark tensorflow regualr compiler\
                                            vs tensorflow xla compiler vs torch vs torch new compiler.')
parser.add_argument('--test', type=ast.literal_eval,default=True, help='would inform the script if this is just a test run or a full run')

args = parser.parse_args()

currentDT = str(datetime.datetime.now()).split('.')[0]

if args.test:
    logfile_path="logs/test_logs.log"
    name_suffix='test'
else:
    logfile_path='logs/run_'+currentDT+'.log'
    name_suffix=currentDT


logging.basicConfig(filename=logfile_path, level=logging.DEBUG)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

batch_size = 16
size = (batch_size, 3, 256, 256)
test_epochs = [10, 15, 30, 150, 300]
epochs = np.ceil(np.logspace(1, 3, 30, endpoint=False)).astype('int')
mu = 0
sigma = 2
size2 = (batch_size, 64)
x_val = np.random.normal(mu, sigma, size) 
y_val = np.random.normal(mu, sigma, size2)
z_val = np.random.normal(mu, sigma, size2)

if args.test:
    epochs = test_epochs

print('Test mode flag set to {}, then : \nNum epochs is : \n    {}\nLogfile path is :\n    {}'.format(args.test, epochs, logfile_path))

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
        
    def forward(self, x, y, z):
        
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

def model_fn(x, y, z):
    x1 = tf.layers.conv2d(
        x,
        filters=32,
        kernel_size=3,
        padding="same",
        name="conv2d/1")

    x3 = tf.layers.flatten(x1)
    logits = tf.layers.dense(x3, units=64, name="dense/1")
    out = tf.nn.elu(logits)
    result = tf.reduce_sum(out + y * z)
    return result

def create_and_run_graph(xla_enabled):
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:

            x = tf.placeholder(tf.float32, shape=(None, 256, 256, 3), name='x')
            y = tf.placeholder(tf.float32, shape=(None, 64), name='y')
            z = tf.placeholder(tf.float32, shape=(None, 64), name='z') 

            
        
            if xla_enabled == True:
                result = xla.compile(computation=model_fn, inputs=(x, y, z))[0]

            else:
                result = model_fn(x, y, z)

            # `result` is a normal Tensor (albeit one that is computed by an XLA
            # compiled executable) and can be used like any other Tensor.
            
            
            sess.run(tf.global_variables_initializer())
            x_val1 = x_val.swapaxes(1,3)
            start = time. time()
            for _ in range(epoch):
                output = sess.run(result, feed_dict={x: x_val1, y: y_val, z: z_val}) # you can add memory info by adding options=run_opts to the sess.run
            end = time. time()
            p_t = end - start
            # retrieve all the variables and delete them manually
            # a = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # print(tf.get_default_graph().get_name_scope())
            # print(a)
            # for x in a:
            #     l = tf.get_variable(x.name, x.shape)
            #     print(l)
            print('maxrss: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print('maxrss: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print('maxrss: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return output, p_t
    


def run_torch():
    x_val_ten, y_val_ten, z_val_ten = (torch.tensor(i).float().cuda() for i in [x_val, y_val, z_val] )
    m = SimpleCNN()
    m.cuda()
    start = time. time()
    for _ in range(epoch):
        out = m.forward(x_val_ten, y_val_ten, z_val_ten)
    end = time. time()
    p_t = end - start
    return out, p_t


txla, t, tt, outxla, out, torchoutput, percentage_graph_TRvTXLA, percentage_graph_TRvP, percentage_graph_PvTXLA, x_graph_TRvTXLA, x_graph_TRvP, x_graph_PvTXLA= ([] for i in range(12))

for epoch in epochs:
    print('##################################### XLA Enabled #############################################')
    outputxla, p_txla = create_and_run_graph(True)
    txla.append(p_txla)
    print('##################################### XLA disabled ############################################')
    output, p_t = create_and_run_graph(False)
    t.append(p_t)
    print('######################################## Pytorch ##############################################')
    # reload(tf)
    # reload(xla)
    # torchoutput, tp = run_torch()
    # tt.append(tp)
    logging.info("epoch num: {}, execution time xla: {}, execution time regular {}".format(epoch, p_txla, p_t))
    print("epoch num: {}, execution time xla: {}, execution time regular {}".format(epoch, p_txla, p_t))
    percentage_graph_TRvTXLA.append(100*(p_t-p_txla)/p_t)
    # percentage_graph_TRvP.append(100*(tp-p_t)/p_t)
    # percentage_graph_PvTXLA.append(100*(tp-p_txla)/tp)

    x_graph_TRvTXLA.append(p_t/p_txla)
    # x_graph_TRvP.append(tp/p_t)
    # x_graph_PvTXLA.append(tp/p_txla)


plt.figure()
plt.xlabel('number of epochs')
plt.ylabel('execution time')
plt.title('TF Regular vs TF XLA vs Pytorch compiler Benchmarking')
plt.plot(epochs, txla, 'r-o',label='xla compiler')
plt.plot(epochs, t, 'b-o', label='regular compiler')
# plt.plot(epochs, tt, 'k-o', label='pytorch compiler')
# plt.yscale('log')
# plt.xscale('log')
plt.grid(True)
plt.legend()
plt.savefig('results/exectime_'+name_suffix)

plt.figure()
plt.xlabel('number of epochs')
plt.ylabel('performance percentage')
plt.title('TF Regular vs TF XLA vs Pytorch percentage gain')
plt.plot(epochs, percentage_graph_TRvTXLA, 'r-o',label='TF_R vs TF_XLA')
# plt.plot(epochs, percentage_graph_TRvP, 'b-o', label='TF_R vs Torch')
# plt.plot(epochs, percentage_graph_PvTXLA, 'k-o', label='Torch vs TF XLA')
plt.legend()
plt.grid(True)
plt.savefig('results/percentage_'+name_suffix)

plt.figure()
plt.xlabel('number of epochs')
plt.ylabel('xtimes gain')
plt.title('TF Regular vs TF XLA vs Pytorch speedup gain')
plt.plot(epochs, x_graph_TRvTXLA, 'r-o',label='TF_R vs TF_XLA')
# plt.plot(epochs, x_graph_TRvP, 'b-o', label='TF_R vs Torch')
# plt.plot(epochs, x_graph_PvTXLA, 'k-o', label='Torch vs TF XLA')
plt.legend()
plt.grid(True)
plt.savefig('results/speedup_'+name_suffix)

#bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.


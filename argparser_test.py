import argparse
import numpy as np
import ast
import datetime

parser = argparse.ArgumentParser(description='Merged test to benchmark tensorflow regualr compiler\
                                            vs tensorflow xla compiler vs torch vs torch new compiler.')
parser.add_argument('--test', type=ast.literal_eval, default=True, help='would inform the script if this is just a test run or a full run')

args = parser.parse_args()

currentDT = datetime.datetime.now()

if args.test:
    logfile_path="logs/test_logs.log"
else:
    logfile_path='logs/run_'+str(currentDT).split('.')[0]+'.log'


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
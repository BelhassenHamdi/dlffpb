'''
reads data from logfile path and plots results in the benchmark figure
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import datetime
import sys

parser = argparse.ArgumentParser(description='Plot creator for benchmarked data')
parser.add_argument('-p', '--paths', type=str, nargs='+', required=True, help='list of names of the tools to be benchmarked')
parser.add_argument('-f', '--fighters', type=str, nargs='*', default=['rabbit', 'turtle'], help='list of names of the tools to be benchmarked')
parser.add_argument('--performance', type=str, default='Speed', help='performance to be benchmarked, has to be attached, eg. Execution_Time')
parser.add_argument('-v', '--variable', type=str, default='kilometer', help='variable used for testing')
parser.add_argument('-s', '--scale', type=str, default=None, help='you can choose "linear", "log", "symlog", "logit"')
parser.add_argument('--axis', type=int, default=2, required='--scale' in sys.argv, help='axis can be 0 for x axis only, 1 for y axis only and 2 for two axis')

args = parser.parse_args()
globals().update(args.__dict__)


def reader(path):
    var, val = ([] for _ in range(2))
    with open(path, 'r') as f:
        for line in f:
            results = line.split(' ')
            var.append(float(results[0]))
            val.append(float(results[1]))
    return var, val

def context(fighters, performance):
    title=''
    for num,name in enumerate(fighters):
        title = title + name
        if num != len(fighters)-1:
            title=title + ' vs '
    title += ' ' + performance + ' Benchmark'
    return title

def figure():
    plt.figure()
    plt.xlabel(variable)
    plt.ylabel(performance)
    title = context(fighters, performance)
    plt.title(title)
    colors = list(mcolors.CSS4_COLORS)
    for i, name in enumerate(fighters):
        color = colors[i]
        match = [s for s in paths if name in s][0]
        print('file {} has been associated to {}'.format(match, name))
        var, val = reader(match)
        plt.plot(var, val, color=color, marker='o', label=name)
    if scale != None:
        if axis==0:
            plt.xscale(scale)
        elif axis==1:
            plt.yscale(scale)
        else:
            plt.xscale(scale)
            plt.yscale(scale)
    plt.grid(True)
    plt.legend()
    currentDT = str(datetime.datetime.now()).split('.')[0]
    plt.savefig('results/'+title+currentDT)

figure()

'''
    This file contains several tensorflow models ready to be used
'''
import tensorflow as tf
from tensorflow.contrib.compiler import xla

class TFmodel():
    def __init__():
        



class expandable_network():
    '''
    this network expands each x iteration, number of iteration can be set manually or using some automatic threshholding
    the expansion is done through adding a layer before the last classification layer build new graph and initialize the first
    layer using previously learned weights. layers adding can be set to bottom first, surface first or random. In a more advanced version
    we will implement an intelligent way to layer adding.
    The network is still an experiment under construction.
    '''
'''
    This file contains several tensorflow models ready to be used
'''
import tensorflow as tf
from tensorflow.contrib.compiler import xla

class TFmodel():
    def __init__():
        pass
    pass
        



class expandable_network():
    '''
    this network expands each x iteration, number of iteration can be set manually or using some automatic threshholding
    the expansion is done through adding a layer before the last classification layer build new graph and initialize the first
    layer using previously learned weights. layers adding can be set to bottom first, surface first or random. In a more advanced version
    we will implement an intelligent way to layer adding.
    The network is still an experiment under construction.
    '''
    pass

class SimpleCnn():
    def __init__():
        pass

    def network_builder():


    def build():
        conv0 = tf.keras.layers.Conv2D(input_data, numfil0, (3,3), strides=(1, 1), padding='same', activation=tf.keras.activations.elu)
        conv1 = tf.keras.layers.Conv2D(conv0, numfil1, (3,3), strides=(1, 1), padding='same', activation=tf.keras.activations.elu)
        conv2 = tf.keras.layers.Conv2D(conv1, numfil2, (3,3), strides=(1, 1), padding='same', activation=tf.keras.activations.elu)
        flat0 = tf.keras.flatten(conv2)
        dense0 = tf.keras.layers.dense(flat0)
        return dense0
    
    def train():
        pass

    def test():
        pass
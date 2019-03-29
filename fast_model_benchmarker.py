'''
this file is a full and fast testing pipeline for experiments. it's based on mnist and cifar 10, cifar 100
the aim of this model is to plug the model easily into the testing pipeline and start experimenting.
Baseline results well be found in baseline reults folders. whenever a new model is tested and full tuned 
we add it's results to hte baseline.
The early and widely known baselines will be results from cifar 10 cifar 100 mnist. Imagenet will be rolled 
to the project soon
'''

import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


# specify dataset
# specify parameters

# build the input data pipeline
# build models plug
# build data visualization and tensorboar
# build data logger
# build result logger
# build model saver
# build benchmarker tool

# start with a mockup of the pipeline in the same file

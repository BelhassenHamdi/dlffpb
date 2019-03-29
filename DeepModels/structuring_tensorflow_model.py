'''
author : danijar hafner
article : https://danijar.com/structuring-your-tensorflow-models/
'''

# Structuring Your TensorFlow Models

# Defining your models in TensorFlow can easily result in one huge wall of code. How to structure 
# your code in a readable and reusable way? For the impatient of you, here is the link to a working 
# example gist. You might also want to take a look at my new post on fast prototyping in TensorFlow, 
# that builds on the idea described here.

# Defining the Compute Graph

# It’s sensible to start with one class per model. What is the interface of that class? Usually, 
# your model connects to some input data and target placeholders and provides operations for training, 
# evaluation, and inference.

class Model:

    def __init__(self, data, target):
        data_size = int(data.get_shape()[1])
        target_size = int(target.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
        incoming = tf.matmul(data, weight) + bias
        self._prediction = tf.nn.softmax(incoming)
        cross_entropy = -tf.reduce_sum(target, tf.log(self._prediction))
        self._optimize = tf.train.RMSPropOptimizer(0.03).minimize(cross_entropy)
        mistakes = tf.not_equal(
            tf.argmax(target, 1), tf.argmax(self._prediction, 1))
        self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @property
    def prediction(self):
        return self._prediction

    @property
    def optimize(self):
        return self._optimize

    @property
    def error(self):
        return self._error

# This is basically, how models are defined in the TensorFlow codebase. However, there are some problems 
# with it. Most notably, the whole graph is define in a single function, the constructor. This is neither 
# particularly readable nor reusable.

# Using Properties

# Just splitting the code into functions doesn’t work, since every time the functions are called, the 
# graph would be extended by new code. Therefore, we have to ensure that the operations are added to 
# the graph only when the function is called for the first time. This is basically lazy-loading.

class Model:

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self._prediction = None
        self._optimize = None
        self._error = None

    @property
    def prediction(self):
        if not self._prediction:
            data_size = int(self.data.get_shape()[1])
            target_size = int(self.target.get_shape()[1])
            weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
            bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
            incoming = tf.matmul(self.data, weight) + bias
            self._prediction = tf.nn.softmax(incoming)
        return self._prediction

    @property
    def optimize(self):
        if not self._optimize:
            cross_entropy = -tf.reduce_sum(self.target, tf.log(self.prediction))
            optimizer = tf.train.RMSPropOptimizer(0.03)
            self._optimize = optimizer.minimize(cross_entropy)
        return self._optimize

    @property
    def error(self):
        if not self._error:
            mistakes = tf.not_equal(
                tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
            self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return self._error

# This is much better than the first example. Your code now is structured into functions that you
# can focus on individually. However, the code is still a bit bloated due to the lazy-loading logic. 
# Let’s see how we can improve on that.

# Lazy Property Decorator

# Python is a quite flexible language. So let me show you how to strip out the redundant code from the 
# last example. We will use a decorator that behaves like @property but only evaluates the function 
# once. It stores the result in a member named after the decorated function (prepended with a prefix) 
# and returns this value on any subsequent calls. If you haven’t used custom decorators yet, you might 
# also want to take a look at this guide.

import functools

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

# Using this decorator, our example simplifies to the code below.

class Model:

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        data_size = int(self.data.get_shape()[1])
        target_size = int(self.target.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
        incoming = tf.matmul(self.data, weight) + bias
        return tf.nn.softmax(incoming)

    @lazy_property
    def optimize(self):
        cross_entropy = -tf.reduce_sum(self.target, tf.log(self.prediction))
        optimizer = tf.train.RMSPropOptimizer(0.03)
        return optimizer.minimize(cross_entropy)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

# Note that we mention the properties in the constructor. This way the full graph is ensured to be
#  defined by the time we run tf.initialize_variables().

# Organizing the Graph with Scopes

# We now have a clean way to define model in code, but the resulting computations graphs are still 
# crowded. If you would visualize the graph, it would contain a lot of interconnected small nodes. 
# The solution would be to wrap the content of each function by a with tf.name_scope('name') or with 
# tf.variable_scope('name'). Nodes would then be grouped together in the graph. But we adjust our 
# previous decorator to do that automatically:

import functools

def define_scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

# I gave the decorator a new name since it has functionality specific to TensorFlow in addition to the 
# lazy caching. Other than that, the model looks identical to the previous one.

# We could go even further an enable the @define_scope decorator to forward arguments to the 
# tf.variable_scope(), for example to define a default initializer for the scope. If you are 
# interested in this check out the full example I put together.

# We can now define models in a structured and compact way that result in organized computation 
# graphs. This works well for me. If you have any suggestions or questions, feel free to use the 
# comment section.

# Updated 2018-06-26: Added link to my post on prototyping in TensorFlow, that introduces an 
# improved version of the decorator idea introduced here.


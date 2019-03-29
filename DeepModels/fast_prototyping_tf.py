'''
autor : Danijar Hafner
article link : https://danijar.com/patterns-for-fast-prototyping-with-tensorflow/
'''


# Patterns for Fast Prototyping with TensorFlow

# TensorFlow is designed as a framework that supports both production and research code. After using 
# TensorFlow for several years and being involved in its development, I collected a few patterns for 
# faster prototyping that I found myself using in many research projects. This post also introduces 
# an alternative to my previous post on structuring models.

# Dictionary with less typing

# Rapid prototyping requires flexible data structures, such as dictionaries. However, in Python that 
# means typing a lot of square brackets and quotes. The following trick defines an attribute dictionary 
# that allows us to address keys as if they were attributes:

class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__


# Since the class inherits from dict, we can still use the object everywhere where we can use a 
# dictionary. The attribute dictionary is useful for example to define experiment configurations:


def define_config():
  config = AttrDict()
  config.layer_sizes = [100, 100]
  config.output_size = 10
  config.learning_rate = 1e-3
  return config

config = define_config()
print(config.learning_rate)


# A more sophisticated version of the attribute dictionary could add a method to freeze its object 
# to prevent accidental changes. If you want more flexibility than that, check out Ian Fischer’s pstar 
# library.

# Decorator to share variables

# Many neural network models reuse parts of the network, for example to apply them to different inputs 
# or to share weights. TensorFlow’s tf.make_template() wraps a function to ensure that all its invokations 
# use the same weights (or raise an error). However, it is a bit clumsy to use, so the following decorator 
# makes it easier:


share_variables = lambda func: tf.make_template(
    func.__name__, func, create_scope_now_=True)


# This allows us to decorate functions to share the variables between all calls of the function. 
# Just place the @share_variables decorator before the function name:


@share_variables
def my_network(inputs, config):
  hidden = tf.layers.flatten(inputs)
  for size in config.layer_sizes:
    hidden = tf.layers.dense(hidden, size, tf.nn.relu)
  prediction = tf.layers.dense(hidden, config.output_size)
  return prediction


# An extended version of this decorator also supports methods of classes, so that each object uses its own 
# weights, but method calls on the same object share weights. I have created a code example of this 
# template decorator.

# Hold the graph together

# A typical problem when writing TensorFlow code is that many quantities exist both as tensors in the 
# graph and as actual values. To avoid confusion among all the objects, it is useful to group the tensors 
# together into a graph object. This also allows to quickly pass the whole graph to a function.


def define_graph(config):
  tf.reset_default_graph()
  inputs, targets = define_data_pipeline()
  prediction = my_network(inputs, config)
  loss = tf.losses.mean_squared_error(targets, prediction)
  optimizer = config.optimizer()
  optimize = optimizer.minimize(loss)
  return AttrDict(locals())  # The magic line.


# The above function defines a lot of tensors for our model. It then uses locals() to access the 
# dictionary of all local variables, and wraps them into an attribute dictionary that we defined earlier. 
# As a result, we can access tensors via the graph object:


graph = define_graph(config)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for _ in range(config.num_epochs):
    sess.run(graph.optimize)
  loss = sess.run(graph.loss)  # No name collision anymore.
  print(loss)

# Conclusion

# I hope these ideas help you to be more productive at prototyping with TensorFlow. Some of the ideas 
# are certainly not well suited for production code. But they allow you to iterate quickly – one of the 
# most important aspects of research. Please feel free to share your thoughts and feedback.
import tensorflow as tf
from tensorflow.contrib.compiler import xla
import logging
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)


batch_size = 32
size = (batch_size, 256, 256, 3)
test_epochs = [10, 15, 20]
epochs = np.ceil(np.logspace(1, 3, 30, endpoint=False)).astype('int')
mu = 0
sigma = 2
size2 = (batch_size, 64)
x_val = np.random.normal(mu, sigma, size) 
y_val = np.random.normal(mu, sigma, size2)
z_val = np.random.normal(mu, sigma, size2)

def model_fn(x, y, z):
    x1 = tf.layers.conv2d(
        x,
        filters=32,
        kernel_size=3,
        padding="same",
        name="conv2d/1")
    # print('=============>> first output size is : ',x1.shape)
    # x2 = tf.layers.conv2d(
    #     x1,
    #     filters=64,
    #     kernel_size=3,
    #     padding="same",
    #     name="conv2d/2")
    # print('=============>> second output size is : ',x2.shape)
    x3 = tf.layers.flatten(x1)
    # print('=============>> third output size is : ',x3.shape)
    logits = tf.layers.dense(x3, units=64, name="dense/1")
    # print('=============>> fourth output size is : ',logits.shape)
    out = tf.nn.elu(logits)
    # print('=============>> fifth output size is : ',out.shape)
    result = tf.reduce_sum(out + y * z)
    # print('=============>> sexth output size is : ',result.shape)
    return result

def create_and_run_graph(xla_enabled):
    # config = tf.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    # tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:

            x = tf.placeholder(tf.float32, shape=(None, 256, 256, 3), name='x')
            y = tf.placeholder(tf.float32, shape=(None, 64), name='y')
            z = tf.placeholder(tf.float32, shape=(None, 64), name='z') 

            if xla_enabled == True:
                result = xla.compile(computation=model_fn, inputs=(x, y, z))[0]
                # result = tf.add(result, result)

            else:
                result = model_fn(x, y, z)

            # `result` is a normal Tensor (albeit one that is computed by an XLA
            # compiled executable) and can be used like any other Tensor.
            
            start = time. time()
            sess.run(tf.global_variables_initializer())
            for _ in range(epoch):
                output = sess.run(result, feed_dict={x: x_val, y: y_val, z: z_val}) # you can add memory info by adding options=run_opts to the sess.run
            end = time. time()
            p_t = end - start
    gc.collect()
    return output, p_t

txla, t, outxla, out, percentage_graph= ([] for i in range(5))

for epoch in epochs:
    print('##################################### XLA Enabled ############################################')
    outputxla, p_txla = create_and_run_graph(True)
    txla.append(p_txla)
    # outxla.append(outputxla)
    print('##################################### XLA disabled ############################################')
    output, p_t = create_and_run_graph(False)
    t.append(p_t)
    # out.append(output)
    logging.info("epoch num: {}, execution time xla: {}, execution time regular {}".format(epoch, p_txla, p_t))
    percentage_graph.append(100*(p_t-p_txla)/p_t)


# epochs = test_epochs
plt.figure()
plt.xlabel('number of epochs')
plt.ylabel('execution time')
plt.title('Regular vs XLA compiler Benchmarking')
plt.plot(epochs, txla, 'r-o')
plt.plot(epochs, t, 'b-o')
plt.savefig('exectime')

# plt.figure()
# plt.plot(epochs, outxla, 'r')
# plt.plot(epochs, out, 'b')
# plt.savefig('vallue_table')

# print('first output : ',output)
# print('second output : ',outputxla)

# if output == outputxla :
#     print('SIMILAR RESULTS')
# else:
#     with tf.Session() as sess:
#         print('different results')
#         sess.run(tf.global_variables_initializer())
#         c = tf.metrics.mean_squared_error(tf.Variable(output), tf.Variable(outputxla))
#         print("output difference is : ", sess.run(c))
# print('processing time with xla is : ', p_txla)
# print('processing time without xla is : ', p_t)
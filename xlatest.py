import tensorflow as tf
from tensorflow.contrib.compiler import xla
import time

def model_fn(x, y, z):
    for i in range(500):
        z = tf.reduce_sum(x + y * z)
    return z

def create_and_run_graph(xla_enabled):
    with tf.Session() as sess:
        x = tf.truncated_normal((40,1000,1000), name='x')
        y = tf.truncated_normal((40,1000,1000), name='y')
        z = tf.truncated_normal((40,1000,1000), name='z')    
    
        if xla_enabled:
            result = xla.compile(computation=model_fn, inputs=(x, y, z))[0]

        else:
            result = model_fn(x, y, z)

        # `result` is a normal Tensor (albeit one that is computed by an XLA
        # compiled executable) and can be used like any other Tensor.
        result = tf.add(result, result) 
        start = time. time()
        output = sess.run(result)
        end = time. time()
        p_t = end - start
        return output, p_t


outputxla, p_txla = create_and_run_graph(True)
output, p_t = create_and_run_graph(False)

print('first output : ',output)
print('second output : ',outputxla)

# if output == outputxla :
#     print('SIMILAR RESULTS')
# else:
#     with tf.Session() as sess:
#         print('different results')
#         sess.run(tf.global_variables_initializer())
#         c = tf.metrics.mean_squared_error(tf.Variable(output), tf.Variable(outputxla))
#         print("output difference is : ", sess.run(c))
print('processing time with xla is : ', p_txla)
print('processing time without xla is : ', p_t)
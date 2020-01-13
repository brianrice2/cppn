import tensorflow as tf
from tensorflow.keras import layers

'''
some helper code borrowed from:
https://github.com/carpedm20/DCGAN-tensorflow
'''

#class Linear(layers.Layer):
#    def __init__(self, units = 32, input_dim = 32, bias_start = 0.0,
#               stddev = 1.0):
#        super(Linear, self).__init__()
#        w_init = tf.random_normal_initializer(stddev=stddev)
#        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
#                                                  dtype='float32'),
#                             trainable=True,
#                             name='Matrix')
#        b_init = tf.constant_initializer(bias_start)
#        self.b = tf.Variable(initial_value=b_init(shape=(units, 1),
#                                                  dtype='float32'),
#                             trainable=True,
#                             name='bias')
#
#    def call(self, inputs, with_w = False):
#        if with_w:
#            return tf.matmul(inputs, self.w) + self.b, self.w, self.b
#        else: 
#            return tf.matmul(inputs, self.w) + self.b


class FullyConnected(layers.Layer):
    def __init__(self, net_size = 32):
        super(FullyConnected, self).__init__()
        self.net_size = net_size
                
    def build(self, input_shape, stddev = 1.0):
        w_init = tf.random_normal_initializer(stddev=stddev)
        self.w = self.add_weight(shape=(input_shape[1], self.net_size),
                                 initializer=w_init,
                                 trainable=False)
        self.b = self.add_weight(shape=(1, self.net_size),
                                 initializer=w_init,
                                 trainable=False)

    def call(self, inputs, with_bias = False):
        shape = inputs.get_shape().as_list()
        result = tf.matmul(inputs, self.w)
        
        if with_bias:
            result += self.b * tf.ones([shape[0], 1], dtype=tf.float32)

        return result


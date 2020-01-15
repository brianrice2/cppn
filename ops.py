import tensorflow as tf
from tensorflow.keras import layers

'''
some helper code borrowed from:
https://github.com/carpedm20/DCGAN-tensorflow
'''

class Linear(layers.Layer):
    def __init__(self, net_size = 32, with_w = False):
        super(Linear, self).__init__()
        self.net_size = net_size
        self.with_w = with_w
    
    def build(self, input_shape, stddev = 1.0):
        w_init = tf.random_normal_initializer(stddev=stddev)
        self.w = self.add_weight(shape=(input_shape[1], self.net_size),
                                 initializer=w_init,
                                 trainable=False)
        
        b_init = tf.constant_initializer(0)
        self.b = self.add_weight(shape=(1, self.net_size),
                                 initializer=b_init,
                                 trainable=False)

    def call(self, inputs, with_w = False):
        self.with_w = with_w
        if self.with_w:
            return tf.matmul(inputs, self.w) + self.b, self.w, self.b
        else: 
            return tf.matmul(inputs, self.w) + self.b


class FullyConnected(layers.Layer):
    def __init__(self, net_size = 32, with_bias = False):
        super(FullyConnected, self).__init__()
        self.net_size = net_size
        self.with_bias = with_bias
                
    def build(self, input_shape, stddev = 1.0):
        w_init = tf.random_normal_initializer(stddev=stddev)
        self.w = self.add_weight(shape=(input_shape[1], self.net_size),
                                 initializer=w_init,
                                 trainable=False)
        
        self.b = self.add_weight(shape=(1, self.net_size),
                                 initializer=w_init,
                                 trainable=False)

    def call(self, inputs, with_bias = False):
        self.with_bias = with_bias
        shape = inputs.get_shape().as_list()
        result = tf.matmul(inputs, self.w)
        
        if self.with_bias:
            result += self.b * tf.ones([shape[0], 1], dtype=tf.float32)

        return result


'''
Implementation of Compositional Pattern Producing Networks in Tensorflow
https://en.wikipedia.org/wiki/Compositional_pattern-producing_network

@hardmaru, 2016
Updated @brianrice2, 2020

See hardmaru's blog post here:
http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/
'''

import numpy as np
import tensorflow as tf
import ops

class CPPN():
    def __init__(self, batch_size=1, x_dim=256, y_dim=256,
                 z_dim=32, c_dim=1, net_size=32, scale=8.0):
        """
        Args:
        batch_size
        z_dim: how many dimensions of the latent space vector (R^z_dim)
        c_dim: 1 for mono, 3 for rgb.  dimension for output space.
               you can modify code to do HSV rather than RGB.
        net_size: number of nodes for each fully connected layer of cppn
        scale: the bigger, the more zoomed out the picture becomes
        """

        self.batch_size = batch_size
        self.net_size = net_size
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.scale = scale
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.n_points = x_dim * y_dim

        # populate coordinate vectors
        self.x_vec, self.y_vec, self.r_vec = self.get_coordinates(x_dim, y_dim, scale)
        
        # build network graph
        self.layer_z = ops.FullyConnected(net_size)
        self.layer_x = ops.FullyConnected(net_size)
        self.layer_y = ops.FullyConnected(net_size)
        self.layer_r = ops.FullyConnected(net_size)
        self.layer_fc = ops.FullyConnected(net_size)
        self.layer_fc_cdim = ops.FullyConnected(self.c_dim)
        
    def get_coordinates(self, x_dim=32, y_dim=32, scale=1.0):
        '''
        Calculates and returns a vector of x and y coordinates,
        and corresponding radius from the center of image.

        This is the function c = f(w,x,y,r), which defines the
        intensity of the image for every point in space.
        '''
        n_points = x_dim * y_dim
        x_range = scale * (np.arange(x_dim) - (x_dim-1)/2.0) / (x_dim-1) / 0.5
        y_range = scale * (np.arange(y_dim) - (y_dim-1)/2.0) / (y_dim-1) / 0.5
        
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt((x_mat * x_mat) + (y_mat * y_mat))
        
        x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
        
        return x_mat, y_mat, r_mat
        
    def generator(self, x_dim, y_dim, reuse=False):
        n_points = x_dim * y_dim
        
        # ensure that tensors are float32
        self.x = tf.cast(self.x, tf.float32)
        self.y = tf.cast(self.y, tf.float32)
        self.z = tf.cast(self.z, tf.float32)
        self.r = tf.cast(self.r, tf.float32)
        
        # note that latent vector z is scaled to self.scale factor
        z_scaled = tf.reshape(self.z, [self.batch_size, 1, self.z_dim]) * \
                        tf.ones([n_points, 1], dtype=tf.float32) * self.scale
        z_unroll = tf.reshape(z_scaled, [self.batch_size*n_points, self.z_dim])
        x_unroll = tf.reshape(self.x, [self.batch_size*n_points, 1])
        y_unroll = tf.reshape(self.y, [self.batch_size*n_points, 1])
        r_unroll = tf.reshape(self.r, [self.batch_size*n_points, 1])
        
        if not reuse:
            # reinitialize fully connected layers
            self.layer_z = ops.FullyConnected(self.net_size)
            self.layer_x = ops.FullyConnected(self.net_size)
            self.layer_y = ops.FullyConnected(self.net_size)
            self.layer_r = ops.FullyConnected(self.net_size)
            self.layer_fc = ops.FullyConnected(self.net_size)
            self.layer_fc_cdim = ops.FullyConnected(self.c_dim)
        
        U_output = self.layer_z(z_unroll, with_bias=True) + self.layer_x(x_unroll) \
                   + self.layer_y(y_unroll) + self.layer_r(r_unroll)
        
        '''
        Below are a bunch of examples of different CPPN configurations.
        Feel free to comment out and experiment!
        '''
        
        ### Example: 3 layers of tanh() layers, with net_size = 32 activations/layer
        ### Higher number of loops produces more complex/detailed images,
        ### but gets to be too much by around 8
        H = tf.nn.tanh(U_output)
        for i in range(5):
            H = tf.nn.tanh(self.layer_fc(H, with_bias=True))
        output = tf.sigmoid(self.layer_fc_cdim(H, with_bias=True))
        
        ### Similar to example above, but instead the output is
        ### a weird function rather than just the sigmoid
        ### Outputs a sort of 'dark-themed' picture
        '''
        H = tf.nn.tanh(U_output)
        for i in range(3):
            H = tf.nn.tanh(self.layer_fc(H, with_bias=True))
        output = tf.sqrt(1.0-tf.abs(tf.tanh(self.layer_fc_cdim(H, with_bias=True))))
        '''
        
        ### Example: mixing softplus and tanh layers, with net_size = 32 activations/layer
        '''
        H = tf.nn.tanh(U_output)
        H = tf.nn.softplus(self.layer_fc(H, with_bias=True))
        H = tf.nn.tanh(self.layer_fc(H, with_bias=True))
        H = tf.nn.softplus(self.layer_fc(H, with_bias=True))
        H = tf.nn.tanh(self.layer_fc(H, with_bias=True))
        H = tf.nn.softplus(self.layer_fc(H, with_bias=True))
        output = tf.sigmoid(self.layer_fc_cdim(H, with_bias=True))
        '''
        
        ### Example: mixing sinusoids, tanh and multiple softplus layers
        '''
        H = tf.nn.tanh(U_output)
        H = tf.nn.softplus(self.layer_fc(H, with_bias=True))
        H = tf.nn.tanh(self.layer_fc(H, with_bias=True))
        H = tf.nn.softplus(self.layer_fc(H, with_bias=True))
        output = 0.5 * tf.sin(self.layer_fc_cdim(H, with_bias=True)) + 0.5
        '''
        
        ### Example: residual network of 4 tanh() layers
        ### This layer produces more soft-formed images
        '''
        H = tf.nn.tanh(U_output)
        for i in range(7):
            H = H + tf.nn.tanh(self.layer_fc(H, with_bias=True))
        output = tf.sigmoid(self.layer_fc_cdim(H, with_bias=True))
        '''
        
        # The final hidden later is passed through a fully connected sigmoid
        # layer, so that outputs are in (0, 1).
        # The output has a dimension of c_dim, so can be monotone or RGB
        result = tf.reshape(output, [self.batch_size, y_dim, x_dim, self.c_dim])
        return result
        
    def generate(self, z=None, x_dim=26, y_dim=26, scale=8.0):
        """
        Generate data by sampling from latent space.
        If z is not None, data for this point in latent space is
        generated. Otherwise, z is drawn from prior in latent space.
        """
        if z is None:
            z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)
        self.z = z
        
        # Note: This maps to mean of distribution
        # We could alternatively sample from Gaussian distribution
        self.x, self.y, self.r = self.get_coordinates(x_dim, y_dim, scale=scale)
        image = self.generator(x_dim=x_dim, y_dim=y_dim, reuse=True)
        return image

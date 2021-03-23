from tensorflow.python.keras import backend as K
import tensorflow as tf
from keras.layers import Layer

# Creds to ykamikawa/tf-keras-SegNet
class MaxPoolingWithArgmax2D(Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = tf.nn.max_pool_with_argmax(
                    inputs,
                    ksize=ksize,
                    strides=strides,
                    padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                    K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        
        mask = tf.cast(mask, 'int32')
        
        input_shape = tf.shape(updates, out_type='int32')

        size_matrix = tf.constant(([1, self.size[0], self.size[1], 1]))
        output_shape = tf.math.multiply(input_shape, size_matrix)
        
        one_like_mask = tf.ones_like(mask, dtype='int32')
        batch_shape = tf.concat(
                [[input_shape[0]], [1], [1], [1]],
                axis=0)
        batch_range = tf.reshape(
                tf.range(output_shape[0], dtype='int32'),
                shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range
        
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(
            tf.stack([b, y, x, f]),
            [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)        

        #Keras cannot read input dependent tensor-shape while constructing the layer
        #tf.keras dropped use of compute_output_shape for custom non-dynamic layers
        output_shape = tf.constant(([-1, updates.shape[1]*self.size[0],
                            updates.shape[2]*self.size[1],
                            updates.shape[3] ]))
        
        return tf.reshape(ret, output_shape)

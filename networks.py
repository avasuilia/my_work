"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from ops import *
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Layer,Input, Dense, Reshape, Flatten, Concatenate, Conv2D, Conv2DTranspose, GlobalAveragePooling2D, UpSampling2D, LeakyReLU, ReLU, Add, Multiply, Lambda, Dot, BatchNormalization, Activation, ZeroPadding2D, Cropping2D, Cropping1D
from tensorflow.keras.initializers import TruncatedNormal, he_normal
import tensorflow.keras.backend as K
# from tensorflow
import numpy as np

#assets

from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape

def l2normalize(v, eps=1e-12):
    return v / (tf.norm(v) + eps)


class ConvSN2D(tf.keras.layers.Conv2D):

    def __init__(self, filters, kernel_size, power_iterations=1, **kwargs):
        super(ConvSN2D, self).__init__(filters, kernel_size, **kwargs)
        self.power_iterations = power_iterations


    def build(self, input_shape):
        super(ConvSN2D, self).build(input_shape)

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        self.u = self.add_weight(self.name + '_u',
            shape=tuple([1, self.kernel.shape.as_list()[-1]]), 
            initializer=tf.initializers.RandomNormal(0, 1),
            trainable=False
        )

    def compute_spectral_norm(self, W, new_u, W_shape):
        for _ in range(self.power_iterations):

            new_v = l2normalize(tf.matmul(new_u, tf.transpose(W)))
            new_u = l2normalize(tf.matmul(new_v, W))
            
        sigma = tf.matmul(tf.matmul(new_v, W), tf.transpose(new_u))
        W_bar = W/sigma

        with tf.control_dependencies([self.u.assign(new_u)]):
          W_bar = tf.reshape(W_bar, W_shape)

        return W_bar


    def call(self, inputs):
        W_shape = self.kernel.shape.as_list()
        W_reshaped = tf.reshape(self.kernel, (-1, W_shape[-1]))
        new_kernel = self.compute_spectral_norm(W_reshaped, self.u, W_shape)
        outputs = self._convolution_op(inputs, new_kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                    outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')
        if self.activation is not None:
            return self.activation(outputs)

        return outputs

class ConvSN2DTranspose(tf.keras.layers.Conv2DTranspose):

    def __init__(self, filters, kernel_size, power_iterations=1, **kwargs):
        super(ConvSN2DTranspose, self).__init__(filters, kernel_size, **kwargs)
        self.power_iterations = power_iterations


    def build(self, input_shape):
        super(ConvSN2DTranspose, self).build(input_shape)

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        self.u = self.add_weight(self.name + '_u',
            shape=tuple([1, self.kernel.shape.as_list()[-1]]), 
            initializer=tf.initializers.RandomNormal(0, 1),
            trainable=False
        )

    def compute_spectral_norm(self, W, new_u, W_shape):
        for _ in range(self.power_iterations):

            new_v = l2normalize(tf.matmul(new_u, tf.transpose(W)))
            new_u = l2normalize(tf.matmul(new_v, W))
            
        sigma = tf.matmul(tf.matmul(new_v, W), tf.transpose(new_u))
        W_bar = W/sigma

        with tf.control_dependencies([self.u.assign(new_u)]):
          W_bar = tf.reshape(W_bar, W_shape)

        return W_bar

    def call(self, inputs):
        W_shape = self.kernel.shape.as_list()
        W_reshaped = tf.reshape(self.kernel, (-1, W_shape[-1]))
        new_kernel = self.compute_spectral_norm(W_reshaped, self.u, W_shape)

        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
          h_axis, w_axis = 2, 3
        else:
          h_axis, w_axis = 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
          out_pad_h = out_pad_w = None
        else:
          out_pad_h, out_pad_w = self.output_padding

        out_height = conv_utils.deconv_output_length(height,
                                                    kernel_h,
                                                    padding=self.padding,
                                                    output_padding=out_pad_h,
                                                    stride=stride_h,
                                                    dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(width,
                                                    kernel_w,
                                                    padding=self.padding,
                                                    output_padding=out_pad_w,
                                                    stride=stride_w,
                                                    dilation=self.dilation_rate[1])
        if self.data_format == 'channels_first':
          output_shape = (batch_size, self.filters, out_height, out_width)
        else:
          output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = array_ops.stack(output_shape)
        outputs = K.conv2d_transpose(
            inputs,
            new_kernel,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if not context.executing_eagerly():
          out_shape = self.compute_output_shape(inputs.shape)
          outputs.set_shape(out_shape)

        if self.use_bias:
          outputs = tf.nn.bias_add(
              outputs,
              self.bias,
              data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
          return self.activation(outputs)
        return outputs  

class DenseSN(Dense):
    def build(self, input_shape):
        super(DenseSN, self).build(input_shape)

        self.u = self.add_weight(self.name + '_u',
            shape=tuple([1, self.kernel.shape.as_list()[-1]]), 
            initializer=tf.initializers.RandomNormal(0, 1),
            trainable=False)
        
    def compute_spectral_norm(self, W, new_u, W_shape):
        new_v = l2normalize(tf.matmul(new_u, tf.transpose(W)))
        new_u = l2normalize(tf.matmul(new_v, W))
        sigma = tf.matmul(tf.matmul(new_v, W), tf.transpose(new_u))
        W_bar = W/sigma
        with tf.control_dependencies([self.u.assign(new_u)]):
          W_bar = tf.reshape(W_bar, W_shape)
        return W_bar
        
    def call(self, inputs):
        W_shape = self.kernel.shape.as_list()
        W_reshaped = tf.reshape(self.kernel, (-1, W_shape[-1]))
        new_kernel = self.compute_spectral_norm(W_reshaped, self.u, W_shape)
        rank = len(inputs.shape)
        if rank > 2:
          outputs = standard_ops.tensordot(inputs, new_kernel, [[rank - 1], [0]])
          if not context.executing_eagerly():
            shape = inputs.shape.as_list()
            output_shape = shape[:-1] + [self.units]
            outputs.set_shape(output_shape)
        else:
          inputs = math_ops.cast(inputs, self._compute_dtype)
          if K.is_sparse(inputs):
            outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, new_kernel)
          else:
            outputs = gen_math_ops.mat_mul(inputs, new_kernel)
        if self.use_bias:
          outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
          return self.activation(outputs)
        return outputs

#Networks Architecture

init = tf.keras.initializers.he_uniform()

def conv2d(layer_input, filters, kernel_size=4, strides=2, padding='same', leaky=True, bnorm=True, sn=True):
  if leaky:
    Activ = LeakyReLU(alpha=0.2)
  else:
    Activ = ReLU()
  if sn:
    d = ConvSN2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=init, use_bias=False)(layer_input)
  else:
    d = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=init, use_bias=False)(layer_input)
  if bnorm:
    d = BatchNormalization()(d)
  d = Activ(d)
  return d

def deconv2d(layer_input, layer_res, filters, kernel_size=4, conc=True, scalev=False, bnorm=True, up=True, padding='same', strides=2):
  if up:
    u = UpSampling2D((1,2))(layer_input)
    u = ConvSN2D(filters, kernel_size, strides=(1,1), kernel_initializer=init, use_bias=False, padding=padding)(u)
  else:
    u = ConvSN2DTranspose(filters, kernel_size, strides=strides, kernel_initializer=init, use_bias=False, padding=padding)(layer_input)
  if bnorm:
    u = BatchNormalization()(u)
  u = LeakyReLU(alpha=0.2)(u)
  if conc:
    u = Concatenate()([u,layer_res])
  return u

#assets end

def Siamese(name,img_size,img_ch):
    inp=Input(shape=(img_size,img_size,img_ch))
    g1 = conv2d(inp, 256, kernel_size=(img_size,3), strides=1, padding='valid', sn=False)
    g2 = conv2d(g1, 256, kernel_size=(1,9), strides=(1,2), sn=False)
    g3 = conv2d(g2, 256, kernel_size=(1,7), strides=(1,2), sn=False)
    g4 = Flatten()(g3)
    g11=Dense(128)(g4)
    return Model(inp,g11,name=name)

def Flatten(x=None, name='flatten'):

    if x is None:
        return tf.keras.layers.Flatten(name=name)
    else :
        return tf.keras.layers.Flatten(name=name)(x)

class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape=None):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name=self.name + '_u',
                                 dtype=tf.float32, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        super(SpectralNormalization, self).build()

    def call(self, inputs, training=None, mask=None):
        self.update_weights()
        output = self.layer(inputs)
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = None

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)

        self.layer.kernel.assign(self.w / sigma)

class FullyConnected(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, sn=False, name='FullyConnected'):
        super(FullyConnected, self).__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self.sn = sn

        if self.sn:
            self.fc = SpectralNormalization(tf.keras.layers.Dense(self.units,
                                                                  kernel_initializer=weight_initializer,
                                                                  kernel_regularizer=weight_regularizer_fully,
                                                                  use_bias=self.use_bias), name='sn_' + self.name)
        else:
            self.fc = tf.keras.layers.Dense(self.units,
                                            kernel_initializer=weight_initializer,
                                            kernel_regularizer=weight_regularizer_fully,
                                            use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        x = Flatten(x)
        x = self.fc(x)

        return x

class AdaIN(tf.keras.layers.Layer):
    def __init__(self, shape, sn=False, epsilon=1e-5, name='AdaIN'):
        super(AdaIN, self).__init__(name=name)
        self.shape = shape
        self.epsilon = epsilon

        self.gamma_fc = FullyConnected(units=tf.experimental.numpy.prod(self.shape[1:]), use_bias=True, sn=sn)
        self.beta_fc = FullyConnected(units=tf.experimental.numpy.prod(self.shape[1:]), use_bias=True, sn=sn)


    def call(self, x_init, training=True, mask=None):
        x, style = x_init
        x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_std = tf.sqrt(x_var + self.epsilon)

        x_norm = ((x - x_mean) / x_std)

        gamma = self.gamma_fc(style)
        beta = self.beta_fc(style)

        self.shape[0]=-1
        gamma = tf.reshape(gamma, shape=self.shape)
        beta = tf.reshape(beta, shape=self.shape)

        x = (1 + gamma) * x_norm + beta

        return x



# def Generator(name,img_size, img_ch,style_dim):
#     inpA=Input(shape=(img_size,img_size,img_ch))
#     inpB=Input(shape=(style_dim,))
#     g0 = tf.keras.layers.ZeroPadding2D((0,1))(inpA)
#     g1 = conv2d(g0, 256, kernel_size=(img_size,3), strides=1, padding='valid')
#     g2 = conv2d(g1, 256, kernel_size=(1,9), strides=(1,2))
#     g3 = conv2d(g2, 256, kernel_size=(1,7), strides=(1,2))
#     #upscaling
#     g4 = deconv2d(g3,g2, 256, kernel_size=(1,7), strides=(1,2), bnorm=False)
#     g5 = AdaIN()([g4,inpB])
#     g6 = deconv2d(g5,g1, 256, kernel_size=(1,9), strides=(1,2), bnorm=False)
#     g7 = AdaIN()([g6,inpB])
#     g8 = ConvSN2DTranspose(1, kernel_size=(img_size,1), strides=(1,1), kernel_initializer=init, padding='valid', activation='tanh')(g7)
#     return Model([inpA,inpB],g8,name=name)


class Generator(tf.keras.Model):
    def __init__(self,name,img_size,img_ch,style_dim):
        super(Generator,self).__init__(name=name)
        self.img_size=img_size
        self.img_ch=img_ch
        self.style_dim=style_dim

    def call(self, x_init, training=True, mask=None):
        inpA=x_init[0]
        inpB=x_init[1]
        g0 = tf.keras.layers.ZeroPadding2D((0,1))(inpA)
        g1 = conv2d(g0, 256, kernel_size=(self.img_size,3), strides=1, padding='valid')
        g2 = conv2d(g1, 256, kernel_size=(1,9), strides=(1,2))
        g3 = conv2d(g2, 256, kernel_size=(1,7), strides=(1,2))
        #upscaling
        g4 = deconv2d(g3,g2, 256, kernel_size=(1,7), strides=(1,2), bnorm=False)
        g5 = AdaIN(g4.output_shape,name='AdaIN1')([g4,inpB])
        g6 = deconv2d(g5,g1, 256, kernel_size=(1,9), strides=(1,2), bnorm=False)
        g7 = AdaIN(g6.output_shape,name='AdaIN2')([g6,inpB])
        g8 = ConvSN2DTranspose(1, kernel_size=(self.img_size,1), strides=(1,1), kernel_initializer=init, padding='valid', activation='tanh')(g7)
        return g8

class MappingNetwork(tf.keras.Model):
    def __init__(self, style_dim=64, hidden_dim=512, num_domains=2, sn=False, name='MappingNetwork'):
        super(MappingNetwork, self).__init__(name=name)
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.sn = sn

        self.shared_layers, self.unshared_layers = self.architecture_init()

    def architecture_init(self):
        layers = []
        layers += [FullyConnected(units=self.hidden_dim, sn=self.sn, name='shared_fc')]
        layers += [Relu()]

        for i in range(3):
            layers += [FullyConnected(units=self.hidden_dim, sn=self.sn, name='shared_fc_' + str(i))]
            layers += [Relu()]

        shared_layers = Sequential(layers)

        layers = []
        unshared_layers = []

        for n_d in range(self.num_domains):
            for i in range(3):
                layers += [FullyConnected(units=self.hidden_dim, sn=self.sn, name='domain_{}_unshared_fc_{}'.format(n_d, i))]
                layers += [Relu()]
            layers += [FullyConnected(units=self.style_dim, sn=self.sn, name='domain_{}_style_fc'.format(n_d))]

            unshared_layers += [Sequential(layers)]

        return shared_layers, unshared_layers

    def call(self, x_init, training=True, mask=None):
        z, domain = x_init

        h = self.shared_layers(z)
        x = []

        for layer in self.unshared_layers:
            x += [layer(h)]

        x = tf.stack(x, axis=1) # [bs, num_domains, style_dim]
        x = tf.gather(x, domain, axis=1, batch_dims=-1)  # [bs, 1, style_dim]
        x = tf.squeeze(x, axis=1)
        # x = x[:, domain, :] # [bs, style_dim]

        return x

class StyleEncoder(tf.keras.Model):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512, sn=False, name='StyleEncoder'):
        super(StyleEncoder, self).__init__(name=name)
        self.img_size = img_size
        self.style_dim = style_dim
        self.num_domains = num_domains
        self.max_conv_dim = max_conv_dim
        self.sn = sn

        self.channels = 2 ** 14 // img_size # if 256 -> 64
        self.repeat_num = int(np.log2(img_size)) - 2 # if 256 -> 6

        self.shared_layers, self.unshared_layers = self.architecture_init()

    def architecture_init(self):
        # shared layers
        ch_in = self.channels
        ch_out = self.channels
        blocks = []

        blocks += [Conv(ch_in, kernel=3, stride=1, pad=1, sn=self.sn, name='init_conv')]

        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.max_conv_dim)

            blocks += [ResBlock(ch_in, ch_out, downsample=True, sn=self.sn, name='resblock_' + str(i))]
            ch_in = ch_out

        blocks += [Leaky_Relu(alpha=0.2)]
        blocks += [Conv(channels=ch_out, kernel=4, stride=1, pad=0, sn=self.sn, name='conv')]
        blocks += [Leaky_Relu(alpha=0.2)]

        shared_layers = Sequential(blocks)

        # unshared layers
        unshared_layers = []

        for n_d in range(self.num_domains):
            unshared_layers += [FullyConnected(units=self.style_dim, sn=self.sn, name='domain_{}_style_fc'.format(n_d))]

        return shared_layers, unshared_layers

    def call(self, x_init, training=True, mask=None):
        x, domain = x_init

        h = self.shared_layers(x)

        x = []

        for layer in self.unshared_layers:
            x += [layer(h)]

        x = tf.stack(x, axis=1) # [bs, num_domains, style_dim]
        x = tf.gather(x, domain, axis=1, batch_dims=-1) # [bs, 1, style_dim]
        x = tf.squeeze(x, axis=1)

        # x = x[:, domain, :] # [bs, style_dim]

        return x

class Discriminator(tf.keras.Model):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512, sn=False, name='Discriminator'):
        super(Discriminator, self).__init__(name=name)

        self.img_size = img_size
        self.num_domains = num_domains
        self.max_conv_dim = max_conv_dim
        self.sn = sn

        self.channels = 2 ** 14 // img_size # if 256 -> 64
        self.repeat_num = int(np.log2(img_size)) - 2 # if 256 -> 6

        self.encoder = self.architecture_init()

    def architecture_init(self):
        ch_in = self.channels
        ch_out = self.channels
        blocks = []

        blocks += [Conv(ch_in, kernel=3, stride=1, pad=1, sn=self.sn, name='init_conv')]

        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.max_conv_dim)

            blocks += [ResBlock(ch_in, ch_out, downsample=True, sn=self.sn, name='resblock_' + str(i))]

            ch_in = ch_out

        blocks += [Leaky_Relu(alpha=0.2)]
        blocks += [Conv(channels=ch_out, kernel=4, stride=1, pad=0, sn=self.sn, name='conv_0')]

        blocks += [Leaky_Relu(alpha=0.2)]
        blocks += [Conv(channels=self.num_domains, kernel=1, stride=1, sn=self.sn, name='conv_1')]

        encoder = Sequential(blocks)

        return encoder

    def call(self, x_init, training=True, mask=None):
        x, domain = x_init

        x = self.encoder(x)
        x = tf.reshape(x, shape=[x.shape[0], -1]) # [bs, num_domains]

        x = tf.gather(x, domain, axis=1, batch_dims=-1) # [bs, 1]
        # x = x[:, domain] # [bs]

        return x

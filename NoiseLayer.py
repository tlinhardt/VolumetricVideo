import tensorflow as tf
import tensorflow.math as m
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
import numpy as np

class NoiseLayer(layers.Layer):
  '''
  Layer that generates a tensor of Gaussian noise the same shape as the input. 
  Does not use the input data in any way.
  '''
  def __init__(self,mean=0,stddev=.1,**kwargs):
    super(NoiseLayer, self).__init__(**kwargs)
    self.mean = mean
    self.stddev=stddev
  def call(self,inputs,training=None):
    return backend.random_normal( 
      shape=array_ops.shape(inputs),
      mean=self.mean,
      stddev=self.stddev,
      dtype=inputs.dtype
    )
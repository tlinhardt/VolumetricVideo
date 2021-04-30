import tensorflow as tf
import tensorflow.math as m
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

class NoiseLayer(layers.Layer):
  def __init__(self,mean=0,stddev=.1,**kwargs):
    super(NoiseLayer, self).__init__(**kwargs)
    self.Noise = layers.GaussianNoise(stddev)
    self.mean = mean
  def call(self,inputs):
    mn = tf.ones(inputs.shape) * self.mean
    x = m.scalar_mul(0,inputs)
    x = m.add(self.Noise(x),mn)
    return x
    




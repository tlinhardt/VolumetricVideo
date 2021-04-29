import tensorflow as tf
import tensorflow.math as m
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

class NoiseLayer(layers.Layer):
  def __init__(self,stddev=.1,**kwargs):
    super(NoiseLayer, self).__init__(**kwargs)
    self.Noise = layers.GaussianNoise(stddev)

  def call(self,inputs):
    x = m.scalar_mul(0,inputs)
    x = self.Noise(x)
    return x
    




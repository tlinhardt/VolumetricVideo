import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

class TrilinearUpSampling3D(layers.Layer):
  """
  Extending https://github.com/tensorflow/tensorflow/blob/5dcfc51118817f27fad5246812d83e5dccdc5f72/tensorflow/compiler/tf2xla/kernels/image_resize_ops.cc
  to 3D using a locked 3D Convolution layer
  Only allows for square input in first two dimensions
  """
  def __init__(self, size:int=2, **kwargs):
    super(TrilinearUpSampling3D, self).__init__(**kwargs)
    self.interpolator = layers.Conv3DTranspose(1,3,strides=[2,2,2], padding='same',kernel_initializer=MakeKernel)

  def call(self,inputs):
    return self.interpolator(inputs)
    
def MakeKernel(shape,**kwargs):
  """ FIlter looks like this with each non-0 term 1/x (2D)
  | 8 4 8 |  | 4 2 4 |  | 8 4 8 |
  | 4 2 4 |  | 2 1 2 |  | 4 2 4 |
  | 8 4 8 |  | 4 2 4 |  | 8 4 8 |
  """
  tensor = np.array([
    [
      [1/8,1/4,1/8],
      [1/4,1/2,1/4],
      [1/8,1/4,1/8]
    ],
    [
      [1/4,1/2,1/4],
      [1/2,1/1,1/2],
      [1/4,1/2,1/4]
    ],
    [
      [1/8,1/4,1/8],
      [1/4,1/2,1/4],
      [1/8,1/4,1/8]
    ]
  ],dtype=np.float32)
  tensor = tensor[...,np.newaxis,np.newaxis]
  return tf.constant(tensor)


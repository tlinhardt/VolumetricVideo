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
    self.padding1 = ReflectivePadding3D()
    self.interpolator = layers.Conv3DTranspose(1,3,strides=[2,2,2],padding='valid',kernel_initializer=MakeKernel)
    self.interpolator.trainable = False
    self.crop = layers.Cropping3D([[2,3],[2,3],[1,2]])

  def call(self,inputs):
    x = self.padding1(inputs)
    x = self.interpolator(x)
    x = self.crop(x)
    return x
    
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


#pylint: disable=no-name-in-module
#pylint: disable=undefined-variable
#pylint: disable=unexpected-keyword-arg
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape

class ReflectivePadding3D(layers.Layer):
  def __init__(self, data_format=None, **kwargs):
    super(ReflectivePadding3D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)          
    self.input_spec = InputSpec(ndim=5)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        dim1 = input_shape[2] + 1
      else:
        dim1 = None
      if input_shape[3] is not None:
        dim2 = input_shape[3] + 1
      else:
        dim2 = None
      if input_shape[4] is not None:
        dim3 = input_shape[4] + 1
      else:
        dim3 = None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        dim1 = input_shape[1] + 1
      else:
        dim1 = None
      if input_shape[2] is not None:
        dim2 = input_shape[2] + 1
      else:
        dim2 = None
      if input_shape[3] is not None:
        dim3 = input_shape[3] + 1
      else:
        dim3 = None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])

  def call(self, inputs):
    return reflective_spatial_3d_padding(inputs, data_format=self.data_format)

  def get_config(self):
    config = {'data_format': self.data_format}
    base_config = super(ReflectivePadding3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

def reflective_spatial_3d_padding(x, data_format=None):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  if data_format == 'channels_first':
    pattern = [[0, 0],[0, 0],[1, 1],[1, 1],[1,0]]
  else:
    pattern = [[0, 0],[1, 1],[1, 1],[1, 1],[0, 0]]
  return array_ops.pad(x, pattern,mode="SYMMETRIC")
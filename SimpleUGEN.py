import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as layers

from functools import partial

class SimpleUgen(k.Model):
  def __init__(self, channels=5, kernel_size=3, name='SimpleU',**kwargs):
    super(SimpleUgen, self).__init__(name=name, **kwargs)
    self.channels = channels
    self.kernel_size = kernel_size


    Conv = partial(layers.Conv3D,kernel_size=self.kernel_size,strides=2,activation=layers.ReLU(),padding='same')
    DeConv = partial(layers.Conv3DTranspose,kernel_size=self.kernel_size,strides=2,activation=layers.ReLU(),padding='same')
    Concat = partial(layers.Concatenate,axis=-1)

    i = iter(range(4))
    self.DownConv = [
      Conv(self.channels*1,name=f'Conv3D_{next(i)}'),
      Conv(self.channels*2,name=f'Conv3D_{next(i)}'),
      Conv(self.channels*3,name=f'Conv3D_{next(i)}'),
      Conv(self.channels*4,name=f'Conv3D_{next(i)}')
    ]    
    i = iter(range(4))
    self.UpConv = [
      DeConv(self.channels*1,name=f'DeConv3D_{next(i)}'),
      DeConv(self.channels*2,name=f'DeConv3D_{next(i)}'),
      DeConv(self.channels*3,name=f'DeConv3D_{next(i)}'),
      DeConv(self.channels*4,name=f'DeConv3D_{next(i)}')
    ]
    i = iter(range(4))
    self.Concatenate = [
      Concat(name=f'Concat_{next(i)}'),
      Concat(name=f'Concat_{next(i)}'),
      Concat(name=f'Concat_{next(i)}'),
      Concat(name=f'Concat_{next(i)}')
    ]

    self.Output = Conv(1,name='Output')

  def call(self, inputs):
    return self.Output(self._construct(0,inputs))

  def _construct(self,level,input):
    a = self.DownConv[level](input)
    n = tf.random.normal(a.shape)
    if level == 3:
      return self.UpConv[level](self.Concatenate[level]([a,n]))
    else:
      return self.UpConv[level](self.Concatenate[level]([a,self._construct(level+1,a),n]))
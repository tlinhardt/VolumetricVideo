import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as layers
from NoiseLayer import NoiseLayer

from functools import partial

class SimpleUGen(k.Model):
  def __init__(self, channels=5, kernel_size=3, noise_stddev=.1, do_batch_norm:bool=False, batch_norm_alpha=.5, name='SimpleU',**kwargs):
    super(SimpleUGen, self).__init__(name=name, **kwargs)
    self.channels = channels
    self.kernel_size = kernel_size
    self.noise_stddev = noise_stddev
    self.batch_norm_alpha = batch_norm_alpha
    self.do_batch_norm = do_batch_norm

    if self.do_batch_norm:
      Activation = lambda *args, **kwargs: k.Sequential([layers.BatchNormalization(momentum=self.batch_norm_alpha),layers.PReLU()])
    else:
      Activation = partial(layers.PReLU)

    Conv = partial(layers.Conv3D,kernel_size=self.kernel_size,strides=2,padding='same')
    DeConv = partial(layers.Conv3DTranspose,kernel_size=self.kernel_size,strides=2,padding='same')
    Concat = partial(layers.Concatenate,axis=1)
    Noise = partial(NoiseLayer,stddev=self.noise_stddev)

    i = iter(range(4))
    self.DownConv = [
      Conv(self.channels*1,strides=(1,2,2),name=f'Conv3D_{next(i)}'),
      Conv(self.channels*2,strides=(1,2,2),name=f'Conv3D_{next(i)}'),
      Conv(self.channels*3,name=f'Conv3D_{next(i)}'),
      Conv(self.channels*4,name=f'Conv3D_{next(i)}')
    ]
    i = iter(range(4))    
    self.DownActivations = [
      Activation(name=f'DownActivation_{next(i)}'),
      Activation(name=f'DownActivation_{next(i)}'),
      Activation(name=f'DownActivation_{next(i)}'),
      Activation(name=f'DownActivation_{next(i)}'),
    ]    
    i = iter(range(4))
    self.UpConv = [
      DeConv(self.channels*1,strides=(1,2,2),name=f'DeConv3D_{next(i)}'),
      DeConv(self.channels*2,strides=(1,2,2),name=f'DeConv3D_{next(i)}'),
      DeConv(self.channels*3,name=f'DeConv3D_{next(i)}'),
      DeConv(self.channels*4,name=f'DeConv3D_{next(i)}')
    ]
    i = iter(range(4))
    self.Noise = [
      Noise(name=f'Noise_{next(i)}'),
      Noise(name=f'Noise_{next(i)}'),
      Noise(name=f'Noise_{next(i)}'),
      Noise(name=f'Noise_{next(i)}')
    ]
    i = iter(range(4))    
    self.Concatenate = [
      Concat(name=f'Concat_{next(i)}'),
      Concat(name=f'Concat_{next(i)}'),
      Concat(name=f'Concat_{next(i)}'),
      Concat(name=f'Concat_{next(i)}')
    ]
    i = iter(range(4))    
    self.UpActivations = [
      Activation(name=f'UpActivation_{next(i)}'),
      Activation(name=f'UpActivation_{next(i)}'),
      Activation(name=f'UpActivation_{next(i)}'),
      Activation(name=f'UpActivation_{next(i)}'),
    ]    
    self.Output = Conv(1,strides=1,name='Output')

  def call(self, inputs):
    return self.Output(self._construct(0,inputs))

  def _construct(self,level,input):
    a = self.DownConv[level](input)
    n = self.Noise[level](a)
    a = self.DownActivations[level](a)
    if level == 3:
      return self.UpActivations[level](self.UpConv[level](self.Concatenate[level]([a,n])))
    else:
      return self.UpActivations[level](self.UpConv[level](self.Concatenate[level]([a,self._construct(level+1,a),n])))
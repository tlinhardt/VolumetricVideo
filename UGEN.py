import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as layers
from NoiseLayer import NoiseLayer

from Trilinear import TrilinearUpSampling3D

class UGen(k.Model):
  '''
  Class that generates the structure of a generator inspired by the modified 
  U-Net as defined by Kayalibay et.al at https://github.com/BRML/CNNbasedMedicalSegmentation
  '''
  def __init__(self, feature_size:int = 3, init_kernel_count:int = 8, do_batchnorm:bool = True, alpha:float = .5, stddev:float = .1, name='UGen', **kwargs):
    '''
    Constructor for the structure generator

    Params:
    shape: Shape of the input stream, should be 4D with 4th value being the color channel count
    feature_size: Width of the feature learning filters in each convolutional layer, probably should be odd
    init_kernel_count: Number of kernels used in the first layer, further layers scale this value
    do_batchnorm: Should Batch Normalization be done
    alpha: momentum for the Batch Normalization layers
    '''
    super(UGen, self).__init__(name=name, **kwargs)

    self.do_bn = do_batchnorm
    self.alpha = alpha
    self.fsize = feature_size
    self.kc    = init_kernel_count
    self.stddev = stddev

    self.Activation = lambda name=None: k.Sequential([layers.BatchNormalization(momentum=self.alpha),layers.PReLU()], name=name) if self.do_bn else layers.PReLU(name=name)   
    #tensorflow doesn't have a functional trilinear interpolator so using this for now
    self.Interpolator = lambda size=2, name=None: TrilinearUpSampling3D(size=size,name=name)
    
    #Descent
    self.conv = layers.Conv3D(filters=self.kc, kernel_size=self.fsize, padding="same",name="InitialConv")

    self.res1, self._r1s = self.Residual_Block(self.kc*2, num=1)
    self.res1_skip = layers.Add(name="ResSkip_1")
    self.res2, self._r2s = self.Residual_Block(self.kc*4, num=2)
    self.res2_skip = layers.Add(name="ResSkip_2")
    self.res3, self._r3s = self.Residual_Block(self.kc*5, num=3)
    self.res3_skip = layers.Add(name="ResSkip_3")
    self._act4 = self.Activation(name='Activation_4')

    #Ascent
    self.noise4 = NoiseLayer(self.stddev,name='Noise_4')
    self.insert_noise = layers.Concatenate(axis=1)

    self.up3 = self.Up_Block(self.kc*4,self.kc*4,num=3)
    self._act3 = self.Activation(name='Activation_3')
    self.noise3 = NoiseLayer(self.stddev,name='Noise_3')
    self.skip3 = layers.Concatenate(axis=1,name='Skip_3')
    self.post_skip3 = self.Post_Skip_Block(self.kc*8, num=3)
    
    self.up2 = self.Up_Block(self.kc*4,self.kc*2,num=2)
    self._act2 = self.Activation(name='Activation_2')
    self.noise2 = NoiseLayer(self.stddev,name='Noise_2')
    self.skip2 = layers.Concatenate(axis=1,name='Skip_2')
    self.post_skip2 = self.Post_Skip_Block(self.kc*4, num=2)

    self.up1 = self.Up_Block(self.kc*2,self.kc,num=1)
    self._act1 = self.Activation(name='Activation_1')
    self.noise1 = NoiseLayer(self.stddev,name='Noise_1')
    self.skip1 = layers.Concatenate(axis=1,name='Skip_1')
    self.post_skip1 = self.Post_Skip_Block(self.kc*2, num=1)
    
    #Combination
    self.condenser_3 = layers.Conv3D(filters=1,kernel_size=1,padding="same", name="Condenser_3")
    self.condenser_2 = layers.Conv3D(filters=1,kernel_size=1,padding="same", name="Condenser_2")
    self.condenser_1 = layers.Conv3D(filters=1,kernel_size=1,padding="same", name="Condenser_1")

    self.interp3 = self.Interpolator(size=2,name="Interpolator_3")
    self.interp2 = self.Interpolator(size=2,name="Interpolator_2")

    self.upsum2 = layers.Add(name="UpSum_2")
    self.upsum1 = layers.Add(name="UpSum_1")

    self.out = layers.ReLU(name="Final_Activation")

  def call(self, inputs):
    #Descent
    ab1 = self.conv(inputs)

    a = self._r1s(ab1)
    b = self.res1(a)
    ab2 = self.res1_skip([a,b])

    a = self._r2s(ab2)
    b = self.res2(a)
    ab3 = self.res2_skip([a,b])

    a = self._r3s(ab3)
    b = self.res3(a)
    ab4 = self.res3_skip([a,b])
    a = self._act4(ab4)

    #Ascent
    n = self.noise4(a)
    a = self.insert_noise([a,n])

    a = self.up3(a)
    ###############
    n = self.noise3(a)
    a = self.skip3([self._act3(ab3),a,n])
    ps3 = self.post_skip3(a)
    
    a = self.up2(ps3)
    ###############
    n = self.noise2(a)
    a = self.skip2([self._act2(ab2),a,n])
    ps2 = self.post_skip2(a)

    a = self.up1(ps2)
    ###############
    n = self.noise1(a)
    a = self.skip1([self._act1(ab1),a,n])
    ps1 = self.post_skip1(a)

    #Combination
    c3 = self.condenser_3(ps3)
    c2 = self.condenser_2(ps2)
    c1 = self.condenser_1(ps1)
    i3 = self.interp3(c3)
    c2 = self.upsum2([i3,c2])
    c1 = self.upsum1([self.interp2(c2),c1])
    return self.out(c1)

  def model(self, shape):
    x = layers.Input(shape=shape)
    return k.Model(inputs=x, outputs=self.call(x))

  def Residual_Block(self, filters:int, num:int):
    res_skip = k.Sequential(name=f"ResBlockStart_{num}")
    res_skip.add(self.Activation())
    res_skip.add(layers.Conv3D(filters=filters,kernel_size=self.fsize,strides=(1,2,2) if num > 2 else 2,padding="same"))

    res = k.Sequential(name=f"ResBlockEnd_{num}")
    res.add(self.Activation())
    res.add(layers.Conv3D(filters=filters,kernel_size=self.fsize,padding="same"))

    return res,res_skip

  def Up_Block(self, f1:int, f2:int, num:int):
    return k.Sequential([
      layers.Conv3D(filters=f1,kernel_size=1,padding="same"),
      self.Activation(),
      layers.Conv3DTranspose(filters=f2,strides=(1,2,2) if num > 2 else 2,kernel_size=self.fsize,padding="same"),
      self.Activation()
    ], name=f"UpBlock_{num}")

  def Post_Skip_Block(self, filters:int, num:int):
    return k.Sequential([
      layers.Conv3D(filters=filters,kernel_size=self.fsize,padding="same"),
      self.Activation()
    ], name=f"PostSkipBlock_{num}")

import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as layers

class UGen():
  def __init__(self,shape,alpha=.5):
    self.alpha = alpha
    self.shape = shape
    #Descent
    self.input = layers.Input(shape=self.shape)    

    x0 = k.Sequential([
        layers.Conv3D(filters=8,kernel_size=3, padding="same"),
        layers.BatchNormalization(momentum=self.alpha),
        layers.PReLU()
      ])(self.input)

    r1 = self.ResDown(16,x0)
    r2 = self.ResDown(32,r1)
    r3 = self.ResDown(64,r2)

    self.noise1 = layers.Input(shape=r3.shape[1:])
    n  = layers.Concatenate(axis=-1)([r3,self.noise1])

    #Ascent
    u1 = self.ResUp(32,32,n)
    self.noise2 = layers.Input(shape=u1.shape[1:])
    m1 = self.Merge(64,r2,u1,self.noise2)

    u2 = self.ResUp(32,16,m1)
    self.noise3 = layers.Input(shape=u2.shape[1:])
    m2 = self.Merge(32,r1,u2,self.noise3)

    u3 = self.ResUp(16,8,m2)
    self.noise4 = layers.Input(shape=u3.shape[1:])
    m3 = self.Merge(16,x0,u3,self.noise4)

    #Combination
    i1 = layers.Conv3D(filters=1,kernel_size=1,padding="same")(m1)
    i2 = layers.Conv3D(filters=1,kernel_size=1,padding="same")(m2)
    i3 = layers.Conv3D(filters=1,kernel_size=1,padding="same")(m3)
    s1 = self.UpSum(i1,i2)
    s2 = self.UpSum(s1,i3)
    f = layers.Conv3D(filters=1,kernel_size=1,padding="same")(s2)
    
    self.inputs = [self.input,self.noise1,self.noise2,self.noise3,self.noise4]
    self.output = layers.ReLU()(f)

  def ResDown(self,f,x):
    a = layers.Conv3D(filters=f,kernel_size=3,strides=(2,2,2),padding="same")(x)
    b = k.Sequential([
        layers.BatchNormalization(momentum=self.alpha),
        layers.PReLU(),
        layers.Conv3D(filters=f, kernel_size=3,padding="same"),
      ])(a)
    ab = layers.Add()([a,b])
    return k.Sequential([
        layers.BatchNormalization(momentum=self.alpha),
        layers.PReLU()
    ])(ab)
    
  def Merge(self,f,a,b,n):
    concat = layers.Concatenate()([a,b,n])
    return k.Sequential([
        layers.Conv3D(filters=f,kernel_size=3,padding="same"),
        layers.BatchNormalization(momentum=self.alpha),
        layers.PReLU()
      ])(concat)

  def ResUp(self,f1,f2,a):
    return k.Sequential([
        layers.Conv3D(filters=f1,kernel_size=1,padding="same"),
        layers.BatchNormalization(momentum=.5),
        layers.PReLU(),
        layers.Conv3DTranspose(filters=f2,strides=2,kernel_size=3,padding="same"),
        layers.BatchNormalization(momentum=.5),
        layers.PReLU()
      ])(a)

  def UpSum(self,a,b):
    return layers.Add()([
      layers.Conv3DTranspose(filters=5,strides=2,kernel_size=3,padding="same")(a),
      b
      ])


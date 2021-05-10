import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import numpy as np

from SimpleUGEN import SimpleUGen
from Discriminator import discriminator

class FeedbackGAN(k.Model):
  def __init__(self,channels=5,kernel_size=3,noise_mean=0.0, noise_stddev=.1, do_batch_norm:bool=False, batch_norm_alpha=.5, name='FeedbackGAN', visualizer=None,use_noise=False,**kwargs):
    super(FeedbackGAN, self).__init__(**kwargs)
    self.channels = channels
    self.G = SimpleUGen(channels,kernel_size,noise_mean,noise_stddev,do_batch_norm,batch_norm_alpha,'Generator',visualizer,use_noise,**kwargs)
    self.D = discriminator(channels+1,kernel_size,name='Discriminator',**kwargs)

    self.Concat = layers.Concatenate(axis=1)

    #metrics
    self.Tricking = k.metrics.FalsePositives(name='generator_tricking_metric')
    self.Failing = k.metrics.FalseNegatives(name='discriminator_failing_metric')

  def train_step(self,data):
    y,_ = data

    x = tf.slice(y,[0,0,0,0,0],[-1,y.shape[1]-1,-1,-1,-1])

    samples = y.shape[0]

    D_ones = tf.ones( (samples,),dtype=tf.dtypes.float32)
    G_ones = tf.ones( (samples,),dtype=tf.dtypes.float32)
    G_zeros = tf.zeros( (samples,),dtype=tf.dtypes.float32)
    
    self.G.trainable = False    
    self.D.trainable = True    
    #Discriminator on Real Data:
    with tf.GradientTape() as tape:
      pred = self.D(y)
      loss = self.compiled_loss(D_ones, pred, regularization_losses=self.losses)    
    self._descent(tape, loss)

    self.Failing.update_state(D_ones, pred)

    #Discriminator on Generated Data:
    with tf.GradientTape() as tape:
      pred = self(x)
      loss = self.compiled_loss(G_zeros, pred, regularization_losses=self.losses) 
    self._descent(tape, loss)
    
    self.G.trainable = True
    self.D.trainable = False
    #Generator
    with tf.GradientTape() as tape:
      pred = self(x)
      loss = self.compiled_loss(G_ones, pred, regularization_losses=self.losses)
    self._descent(tape, loss)

    self.Tricking.update_state(G_zeros, pred)

    return {m.name: m.result() for m in self.metrics}

  @property
  def Generator(self): return self.G

  def _descent(self,tape, loss):
    #compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

  def call(self,inputs):
    return self.D(self.Concat([inputs,self.G(inputs)]))

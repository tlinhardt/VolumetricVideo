import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import numpy as np

from SimpleUGEN import SimpleUGen

class FeedbackGenerator(SimpleUGen):
  def __init__(self, multi_descent=True, **kwargs):
    super(FeedbackGenerator, self).__init__(**kwargs)

    #choose training method
    if multi_descent:
      self.train_step = self._multi_train_step
    else:
      self.train_step = self._single_train_step

  def _single_train_step(self, data):
    '''
    Meant to be an alternate way to do a feedback loop, never got it fully working so dead code
    '''
    x_set, y_set = data
    x_set = tf.convert_to_tensor(x_set, np.float32)
    y_set = tf.convert_to_tensor(y_set, np.float32)
    #y = tf.convert_to_tensor(y_set[:,i,...], np.float32)
    #preds = np.zeros(x_set[:,:,0:1,...].shape,dtype=np.float32)
    with tf.GradientTape() as tape:
      for i in range(y_set.shape[1]):      
        y = tf.slice(y_set,[0,i,0,0,0],[-1,1,-1,-1,-1])
        pred = self(x_set, training=True)
        x_set = tf.concat([tf.slice(x_set,[0,1,0,0,0],[-1,-1,-1,-1,-1]),pred],1)

      loss = self.compiled_loss(y, pred, regularization_losses=self.losses)
      #compute gradients
      trainable_vars = self.trainable_variables
      #print(tf.convert_to_tensor(preds[:,i,...],np.float32))
      gradients = tape.gradient(loss, trainable_vars)
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))
      self.compiled_metrics.update_state(y, tf.convert_to_tensor(pred,np.float32))

    return {m.name: m.result() for m in self.metrics}

  def _multi_train_step(self, data):
    '''
    Implementation of the feedback optimization
    Does back propagation for multiple frames into the future individually, 
      weighting each one linearly by the distance into the future it is
    '''
    x_set, y_set = data
    x_set = tf.convert_to_tensor(x_set, np.float32)
    y_set = tf.convert_to_tensor(y_set, np.float32)

    for i in range(y_set.shape[1]):
      with tf.GradientTape() as tape:
        y = tf.slice(y_set,[0,i,0,0,0],[-1,1,-1,-1,-1])
        pred = self(x_set, training=True)
        loss = self.compiled_loss(y, pred, regularization_losses=self.losses) * i
      x_set = tf.concat([tf.slice(x_set,[0,1,0,0,0],[-1,-1,-1,-1,-1]),pred],1)

      #compute gradients and back propagate
      trainable_vars = self.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))
      self.compiled_metrics.update_state(y, pred)

    #loss metric number returned is the sum of the frame losses without the weighting
    return {m.name: m.result() for m in self.metrics}
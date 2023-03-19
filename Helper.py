import numpy as np
import wandb

class Helper:
  def sigmoid(self,layer,d=False):
      if(d==True):
          return self.sigmoid(layer)*(1-self.sigmoid(layer))
      return 1.0/(1.0+np.exp(-1.0*layer))

  def identity(self,layer,d=False):
      if(d==True):
          return np.ones(layer.shape)
      return layer

  def softmax(self,layer,d=False):
      if(d==True):
          s=self.softmax(layer,False);
          return s*(1-s)
      newlayer=(layer-np.max(layer))
      return np.exp(newlayer)/np.sum(np.exp(newlayer))

  def cross_entropy(self,true_output,output):
      return -1.0*np.sum(true_output * np.log(output+1e-9))
  
  def mean_square_error(self,true_output,output):
      return np.sum((true_output-output)**2)

  def tanh(self,layer,d=False):
      if(d==True):
          return 1-np.tanh(layer)**2
      return np.tanh(layer)

  def relu(self,layer,d=False):
      if(d==True):
        return 1. * (layer > 0)
      return layer * (layer > 0)

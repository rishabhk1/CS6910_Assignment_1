import numpy as np
import wandb
from Helper import *;
'''
Each class in optimizer.py has same structure i.e constructor and optimize function
optimize function returns the updates weights and bias
'''

'''
class for stochastic gradient descent
optimize function returns weights,bias
'''
class sgd:
  def optimize(self,num_hidden_layers,weights,bias,learning_rate,d_weights,d_bias):
    for i in range(1,num_hidden_layers+2):
      weights[i]=weights[i]-learning_rate*d_weights[i]
      bias[i]=bias[i]-learning_rate*d_bias[i]
    return weights,bias

'''
class for Momentum
return optimize functions weights,bias
'''
class momentum:
  def __init__(self,neurons_list):
    self.update_w={}
    self.update_b={}
    for i in range(len(neurons_list)-1):
      self.update_w[i+1]=np.zeros((neurons_list[i+1],neurons_list[i]))
      self.update_b[i+1]=np.zeros((neurons_list[i+1],1))
  def optimize(self,num_hidden_layers,weights,bias,learning_rate,d_weights,d_bias,beta):
    for i in range(1,num_hidden_layers+2):
        self.update_w[i]=beta*self.update_w[i]+d_weights[i]
        self.update_b[i]=beta*self.update_b[i]+d_bias[i]

    for i in range(1,num_hidden_layers+2):
      weights[i]=weights[i]-learning_rate*self.update_w[i]
      bias[i]=bias[i]-learning_rate*self.update_b[i]

    return weights,bias

'''
class for Nestrov accelerated gradient descent
optimize function returns weights,bias,lookahead weights,lookahead bias
'''
class nag:
  def __init__(self,neurons_list,weights,bias):
    self.update_w={}
    self.update_b={}
    self.weights_use={}
    self.bias_use={}
    for i in range(len(neurons_list)-1):
      self.update_w[i+1]=np.zeros((neurons_list[i+1],neurons_list[i]))
      self.update_b[i+1]=np.zeros((neurons_list[i+1],1))
    self.bias_use=bias.copy()
    self.weights_use=weights.copy()
  
  def optimize(self,num_hidden_layers,weights,bias,learning_rate,d_weights,d_bias,beta):
    for i in range(1,num_hidden_layers+2):
        self.update_w[i]=beta*self.update_w[i]+d_weights[i]
        self.update_b[i]=beta*self.update_b[i]+d_bias[i]

    for i in range(1,num_hidden_layers+2):
      self.weights_use[i]=self.weights_use[i]-learning_rate*self.update_w[i]
      self.bias_use[i]=self.bias_use[i]-learning_rate*self.update_b[i]

    for i in range(1,num_hidden_layers+2):
      weights[i]=self.weights_use[i]-beta*self.update_w[i]
      bias[i]=self.bias_use[i]-beta*self.update_b[i] 

    return weights,bias,self.weights_use,self.bias_use

'''
class for RMSprop
optimize function returns weights,bias
'''
class rmsprop:
  def __init__(self,neurons_list):
    self.update_w={}
    self.update_b={}
    self.epsilon=1e-6
    for i in range(len(neurons_list)-1):
      self.update_w[i+1]=np.zeros((neurons_list[i+1],neurons_list[i]))
      self.update_b[i+1]=np.zeros((neurons_list[i+1],1))

  def optimize(self,num_hidden_layers,weights,bias,learning_rate,d_weights,d_bias,beta):
    for i in range(1,num_hidden_layers+2):
        self.update_w[i]=beta*self.update_w[i]+(1-beta)*((d_weights[i])**2)
        self.update_b[i]=beta*self.update_b[i]+(1-beta)*((d_bias[i])**2)

    for i in range(1,num_hidden_layers+2):
      weights[i]=weights[i]-learning_rate*d_weights[i]/(np.sqrt(self.update_w[i])+self.epsilon)
      bias[i]=bias[i]-learning_rate*d_bias[i]/(np.sqrt(self.update_b[i])+self.epsilon)
      
    return weights,bias



'''
class for Adam
optimize function return weights,bias
'''
class adam:
  def __init__(self,neurons_list):
    self.update_w={}
    self.update_b={}
    self.update_what={}
    self.update_bhat={}
    self.momentum_w={}
    self.momentum_b={}
    self.momentum_what={}
    self.momentum_bhat={}
    self.epsilon=1e-6
    for i in range(len(neurons_list)-1):
      self.update_w[i+1]=np.zeros((neurons_list[i+1],neurons_list[i]))
      self.update_b[i+1]=np.zeros((neurons_list[i+1],1))
      self.momentum_w[i+1]=np.zeros((neurons_list[i+1],neurons_list[i]))
      self.momentum_b[i+1]=np.zeros((neurons_list[i+1],1))
  def optimize(self,num_hidden_layers,weights,bias,learning_rate,d_weights,d_bias,beta1,beta2,t):
    for i in range(1,num_hidden_layers+2):
        self.momentum_w[i]=beta1*self.momentum_w[i]+(1-beta1)*((d_weights[i]))
        self.momentum_what[i]=self.momentum_w[i]/(1-beta1**t)

        self.momentum_b[i]=beta1*self.momentum_b[i]+(1-beta1)*((d_bias[i]))
        self.momentum_bhat[i]=self.momentum_b[i]/(1-beta1**t)

        self.update_w[i]=beta2*self.update_w[i]+(1-beta2)*((d_weights[i])**2)
        self.update_what[i]=self.update_w[i]/(1-beta2**t)

        self.update_b[i]=beta2*self.update_b[i]+(1-beta2)*((d_bias[i])**2)
        self.update_bhat[i]=self.update_b[i]/(1-beta2**t)
  

    for i in range(1,num_hidden_layers+2):
      weights[i]=weights[i]-learning_rate*self.momentum_what[i]/(np.sqrt(self.update_what[i])+self.epsilon)
      bias[i]=bias[i]-learning_rate*self.momentum_bhat[i]/(np.sqrt(self.update_bhat[i])+self.epsilon)
      
    return weights,bias
    
    
'''
class for Nadam
optimize function return weights,bias
'''
class nadam:
  def __init__(self,neurons_list):
    self.update_w={}
    self.update_b={}
    self.update_what={}
    self.update_bhat={}
    self.momentum_w={}
    self.momentum_b={}
    self.momentum_what={}
    self.momentum_bhat={}
    self.epsilon=1e-6
    for i in range(len(neurons_list)-1):
      self.update_w[i+1]=np.zeros((neurons_list[i+1],neurons_list[i]))
      self.update_b[i+1]=np.zeros((neurons_list[i+1],1))
      self.momentum_w[i+1]=np.zeros((neurons_list[i+1],neurons_list[i]))
      self.momentum_b[i+1]=np.zeros((neurons_list[i+1],1))
  def optimize(self,num_hidden_layers,weights,bias,learning_rate,d_weights,d_bias,beta1,beta2,t):
    for i in range(1,num_hidden_layers+2):
        self.momentum_w[i]=beta1*self.momentum_w[i]+(1-beta1)*((d_weights[i]))
        self.momentum_what[i]=self.momentum_w[i]/(1-beta1**t)

        self.momentum_b[i]=beta1*self.momentum_b[i]+(1-beta1)*((d_bias[i]))
        self.momentum_bhat[i]=self.momentum_b[i]/(1-beta1**t)

        self.update_w[i]=beta2*self.update_w[i]+(1-beta2)*((d_weights[i])**2)
        self.update_what[i]=self.update_w[i]/(1-beta2**t)

        self.update_b[i]=beta2*self.update_b[i]+(1-beta2)*((d_bias[i])**2)
        self.update_bhat[i]=self.update_b[i]/(1-beta2**t)
  

    for i in range(1,num_hidden_layers+2):
      weights[i]=weights[i]-learning_rate/(np.sqrt(self.update_what[i])+self.epsilon)*(beta1*self.momentum_what[i]+(1-beta1)*d_weights[i]/(1-beta1**t))
      bias[i]=bias[i]-learning_rate/(np.sqrt(self.update_bhat[i])+self.epsilon)*(beta1*self.momentum_bhat[i]+(1-beta1)*d_bias[i]/(1-beta1**t))
      
    return weights,bias
  





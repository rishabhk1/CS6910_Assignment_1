from Helper import *;
from Optimizer import *
import numpy as np
import wandb

class NeuralNetwork:
  actobj=Helper()
  def __init__(self,num_hidden_layers=3,num_neurons_in_hidden_layer=64,learning_rate=0.001,epoch=10,batch=64,activation="tanh",beta=0.9,beta1=0.999,optimizer="rmsprop",init_strat="xavier",weight_decay=0,loss_type="cross_entropy"):
    self.num_hidden_layers=num_hidden_layers
    self.num_neurons_in_hidden_layer=num_neurons_in_hidden_layer
    self.learning_rate=learning_rate
    self.epoch=epoch
    self.batch=batch
    self.pre_activation_layer={}
    self.activation_layer={}
    self.weights={}
    self.bias={}
    self.d_pre_activation_layer={}
    self.d_activation_layer={}
    self.d_weights={}
    self.d_bias={}
    self.beta=beta
    self.beta1=beta1
    self.optimizer=optimizer
    self.init_strat=init_strat
    self.weight_decay=weight_decay
    self.loss_type=loss_type
    self.activation_fn=activation
    if(activation=="sigmoid"):self.activation_fn=NeuralNetwork.actobj.sigmoid
    if(activation=="tanh"):self.activation_fn=NeuralNetwork.actobj.tanh
    if(activation=="relu"):self.activation_fn=NeuralNetwork.actobj.relu
    if(activation=="identity"):self.activation_fn=NeuralNetwork.actobj.identity
    if(loss_type=="cross_entropy"): self.loss_fn=NeuralNetwork.actobj.cross_entropy
    if(loss_type=="mean_square_error"): self.loss_fn=NeuralNetwork.actobj.mean_square_error

  def weight_initialization(self):
    self.neurons_list=[self.input_size]

    for i in range(self.num_hidden_layers):
      self.neurons_list.append(self.num_neurons_in_hidden_layer)
    self.neurons_list.append(self.output_size)

    for i in range(len(self.neurons_list)-1):
      if(self.init_strat=="random"):
        self.weights[i+1]=np.random.default_rng().uniform(-1,1,(self.neurons_list[i+1],self.neurons_list[i]))
        self.bias[i+1]=np.random.default_rng().uniform(-1,1,(self.neurons_list[i+1],1))
      elif(self.init_strat=="xavier"):
        self.weights[i+1]=np.random.randn(self.neurons_list[i+1],self.neurons_list[i])*np.sqrt(2/(self.neurons_list[i+1]+self.neurons_list[i]))
        self.bias[i+1]=np.random.randn(self.neurons_list[i+1],1)*np.sqrt(2/(self.neurons_list[i+1]+1))
  
  def forward_propogation(self,x):
    self.activation_layer[0]=x
    for i in range(1,self.num_hidden_layers+1):
      self.pre_activation_layer[i]=self.bias[i]+np.matmul(self.weights[i],self.activation_layer[i-1])
      self.activation_layer[i]=self.activation_fn(self.pre_activation_layer[i])
    self.pre_activation_layer[self.num_hidden_layers+1]=self.bias[self.num_hidden_layers+1]+np.matmul(self.weights[self.num_hidden_layers+1],self.activation_layer[self.num_hidden_layers])
    self.activation_layer[self.num_hidden_layers+1]=NeuralNetwork.actobj.softmax(self.pre_activation_layer[self.num_hidden_layers+1])    
    return self.activation_layer[self.num_hidden_layers+1]

  def back_propogation(self,true_output,output):
    d_pre_activation_layer={}
    d_weights={}
    d_activation_layer={}
    d_bias={}
    if(self.loss_type=="cross_entropy"):
      d_pre_activation_layer[self.num_hidden_layers+1]=-1*(true_output-output)
    elif(self.loss_type=="mean_square_error"):
      d_pre_activation_layer[self.num_hidden_layers+1]=-2*(true_output-output)*(output*(1-output))
    for i in range(self.num_hidden_layers+1,0,-1):
      d_weights[i]=np.outer(d_pre_activation_layer[i],self.activation_layer[i-1].T)
      d_bias[i]=d_pre_activation_layer[i]
      if(i==1):break
      d_activation_layer[i-1]=np.matmul(self.weights[i].T,d_pre_activation_layer[i])
      d_pre_activation_layer[i-1]=np.multiply(d_activation_layer[i-1],self.activation_fn(self.pre_activation_layer[i-1],True))#element multiplication

    return d_weights,d_bias,self.loss_fn(true_output,output)
    

  def flush_gradients(self):
    for i in range(len(self.neurons_list)-1):
      self.d_weights[i+1]=np.zeros((self.neurons_list[i+1],self.neurons_list[i]))
      self.d_bias[i+1]=np.zeros((self.neurons_list[i+1],1))

  def accuracy(self,images,labels):
    count=0
    loss=0
    for i in range(images.shape[0]):
      output=self.forward_propogation(images[i].reshape((-1,1)))
      true_output=np.zeros((self.output_size,1))
      true_output[labels[i]]=1
      if(labels[i]==np.argmax(output)):
        count+=1
      loss+=self.loss_fn(true_output,output)
    
    for i in range(1,self.num_hidden_layers+2):
      loss+=(self.weight_decay/2)*np.sum(np.square(self.weights[i]))

    return count*1.0/images.shape[0], loss/images.shape[0]
    

  def run(self,train_images,train_labels,validation_images,validation_labels):
    self.input_size=train_images.shape[1]*train_images.shape[2]
    self.output_size=10
    self.weight_initialization()

    if self.optimizer=="sgd":
      opt=sgd()
    elif self.optimizer=="momentum":
      opt=momentum(self.neurons_list)
    elif self.optimizer=="rmsprop":
      opt=rmsprop(self.neurons_list)
    elif self.optimizer=="nag":
      opt=nag(self.neurons_list,self.weights,self.bias)
    elif self.optimizer=="adam":
      opt=adam(self.neurons_list)
    elif self.optimizer=="nadam":
      opt=nadam(self.neurons_list)

    t=0
    for epoch_no in range(self.epoch):
      val_loss,tra_loss,tra_acc,val_acc=0,0,0,0
      loss=0
      batch_count=0
      self.flush_gradients()
      for i in range(train_images.shape[0]):
        batch_count+=1
        true_output=np.zeros((self.output_size,1))
        true_output[train_labels[i]]=1
        output=self.forward_propogation(train_images[i].reshape((-1,1)))
        d_weights,d_bias,loss=self.back_propogation(true_output,output)
        for j in range(1,self.num_hidden_layers+2):
            self.d_weights[j]=self.d_weights[j]+d_weights[j]
            self.d_bias[j]=self.d_bias[j]+d_bias[j]


        if(batch_count==self.batch):
          for j in range(1,self.num_hidden_layers+2):
            self.d_weights[j]=self.d_weights[j]+self.weight_decay*self.weights[j]

          self.d_weights = {k: v / self.batch for k, v in self.d_weights.items()}
          self.d_bias = {k: v / self.batch for k, v in self.d_bias.items()}
          batch_count=0
          if self.optimizer=="sgd":
            self.weights,self.bias=opt.optimize(self.num_hidden_layers,self.weights,self.bias,self.learning_rate,self.d_weights,self.d_bias)
          elif self.optimizer=="momentum":
            self.weights,self.bias=opt.optimize(self.num_hidden_layers,self.weights,self.bias,self.learning_rate,self.d_weights,self.d_bias,self.beta)
          elif self.optimizer=="rmsprop":
            self.weights,self.bias=opt.optimize(self.num_hidden_layers,self.weights,self.bias,self.learning_rate,self.d_weights,self.d_bias,self.beta)
          elif self.optimizer=="adam":
            t+=1
            self.weights,self.bias=opt.optimize(self.num_hidden_layers,self.weights,self.bias,self.learning_rate,self.d_weights,self.d_bias,self.beta,self.beta1,t)
          elif self.optimizer=="nadam":
            t+=1
            self.weights,self.bias=opt.optimize(self.num_hidden_layers,self.weights,self.bias,self.learning_rate,self.d_weights,self.d_bias,self.beta,self.beta1,t)
          elif self.optimizer=="nag":
            self.weights,self.bias,self.weights_use,self.bias_use=opt.optimize(self.num_hidden_layers,self.weights,self.bias,self.learning_rate,self.d_weights,self.d_bias,self.beta)

          self.flush_gradients()

      if(batch_count>0):#remaining
          for j in range(1,self.num_hidden_layers+2):
            self.d_weights[j]=self.d_weights[j]+self.weight_decay*self.weights[j]

          self.d_weights = {k: v / batch_count for k, v in self.d_weights.items()}
          self.d_bias = {k: v / batch_count for k, v in self.d_bias.items()}
          batch_count=0
          if self.optimizer=="sgd":
            self.weights,self.bias=opt.optimize(self.num_hidden_layers,self.weights,self.bias,self.learning_rate,self.d_weights,self.d_bias)
          elif self.optimizer=="momentum":
            self.weights,self.bias=opt.optimize(self.num_hidden_layers,self.weights,self.bias,self.learning_rate,self.d_weights,self.d_bias,self.beta)
          elif self.optimizer=="rmsprop":
            self.weights,self.bias=opt.optimize(self.num_hidden_layers,self.weights,self.bias,self.learning_rate,self.d_weights,self.d_bias,self.beta)
          elif self.optimizer=="adam":
            t+=1
            self.weights,self.bias=opt.optimize(self.num_hidden_layers,self.weights,self.bias,self.learning_rate,self.d_weights,self.d_bias,self.beta,self.beta1,t)
          elif self.optimizer=="nadam":
            t+=1
            self.weights,self.bias=opt.optimize(self.num_hidden_layers,self.weights,self.bias,self.learning_rate,self.d_weights,self.d_bias,self.beta,self.beta1,t)
          elif self.optimizer=="nag":
            self.weights,self.bias,self.weights_use,self.bias_use=opt.optimize(self.num_hidden_layers,self.weights,self.bias,self.learning_rate,self.d_weights,self.d_bias,self.beta)

          self.flush_gradients()     

      val_acc,val_loss=self.accuracy(validation_images,validation_labels)
      tra_acc,tra_loss=self.accuracy(train_images,train_labels) 
      print(epoch_no+1,'Training loss',tra_loss,'Val loss',val_loss,'Training Accuracy',tra_acc,'Val Accuracy',val_acc)
      wandb.log({"Training loss":tra_loss,'Val loss':val_loss,'Training Accuracy':tra_acc,'Val Accuracy':val_acc})
    if self.optimizer=="nag":
      self.weights=self.weights_use #for nag
      self.bias=self.bias_use #for nag

# CS6910_Assignment_1
### Rishabh Kawediya CS22M072

DLAssignment1 contains Question 1,2,3,7,10.<br>
Sweep contains Question 4,5,6

**Helper.py**
- Activation Functions
    - sigmoid
    - relu
    - tanh
    - identity
- Loss Functions
    - cross entropy
    - mean square error

**To add a new activaton or loss function**

Add below function in  Helper class 
```
def newfunc(layer,d=False):
    if(d==True):
        Compute derivative
    else
        Compute new function
```
    
**Optimizer.py**
  - sgd
  - adam
  - nadam
  - nag
  - momentum
  - rmsprop
  
**To add a new optimizer**<br>
Add below class
```
class newopt:
    def __init__():
    def optimize():
        update rule
        return weights,bias
```
**NeuralNetwork.py**
- NeuralNetwork class
    - forward propogation
    - backward propogation
    - run
    - weight initialization

To create a NeuralNetwork model
```
nn=NeuralNetwork(num_layers,
                    hidden_size,
                    learning_rate,
                    epochs,
                    batch_size,
                    activation,
                    beta,
                    beta2,
                    optimizer, 
                    weight_init,
                    weight_decay,
                    loss)
```
To train model
```
nn.run(train_images,train_labels,validation_images,validation_labels)
```

**To evaluate**

'''
python train.py
'''
  
    
   




# CS6910_Assignment_1
### Rishabh Kawediya CS22M072

DLAssignment1 contains Question 1,2,3,7,10.<br>
Sweep contains Question 4,5,6
## **Packages**
- wandb
- numpy
- tensorflow

## **Helper.py**
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
    
## **Optimizer.py**
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
## **NeuralNetwork.py**
- NeuralNetwork class
    - forward propogation
    - backward propogation
    - run
    - weight initialization

## *To create a NeuralNetwork model*
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
## *To train model*
```
nn.run(train_images,train_labels,validation_images,validation_labels)
```

## **To evaluate**

```
python train.py --wandb_entity myname --wandb_project myprojectname
```


| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.1 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | random | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 4 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | sigmoid | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

<br>




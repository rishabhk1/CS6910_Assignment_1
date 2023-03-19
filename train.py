import argparse
from tensorflow import keras
import numpy as np
import wandb
from Helper import *
from Optimizer import *
from NeuralNetwork import *

parser=argparse.ArgumentParser()
parser.add_argument("-wp","--wandb_project",type=str,default="DLCS6910")
parser.add_argument("-we","--wandb_entity",type=str,default="cs22m072")
parser.add_argument("-d","--dataset",type=str,default="fashion_mnist")
parser.add_argument("-e","--epochs",type=int,default=10)
parser.add_argument("-b","--batch_size",type=int,default=64)
parser.add_argument("-m","--momentum",type=float,default=0.5)
parser.add_argument("-l","--loss",type=str,default="cross_entropy")
parser.add_argument("-o","--optimizer",type=str,default="rmsprop")
parser.add_argument("-lr","--learning_rate",type=float,default=0.001)
parser.add_argument("-beta","--beta",type=float,default=0.9)
parser.add_argument("-beta1","--beta1",type=float,default=0.9)
parser.add_argument("-beta2","--beta2",type=float,default=0.999)
parser.add_argument("-eps","--epsilon",type=float,default=1e-6)
parser.add_argument("-w_d","--weight_decay",type=float,default=0)
parser.add_argument("-w_i","--weight_init",type=str,default="xavier")
parser.add_argument("-nhl","--num_layers",type=int,default=3)
parser.add_argument("-sz","--hidden_size",type=int,default=64)
parser.add_argument("-a","--activation",type=str,default="tanh")
args=parser.parse_args()




wandb.login()
wandb.init(args.wandb_project,args.wandb_entity)

if args.dataset=="fashion_mnist":
	data = keras.datasets.fashion_mnist
else:
	data = keras.datasets.mnist

if args.optimizer in ["adam","nadam"]:
  args.beta=args.beta1
if args.loss !="cross_entropy":
  args.loss="mean_square_error"
if args.activation=="ReLU":
  args.activation="relu"
if args.weight_init=="Xavier":
  args.weight_init="xavier"

(train_images, train_labels), (test_images, test_labels) = data.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
idx = np.arange(train_images.shape[0])
np.random.shuffle(idx)
train_images=train_images[idx]
train_labels=train_labels[idx]
validation_images=train_images[:6000]
validation_labels=train_labels[:6000]
train_images=train_images[6000:]
train_labels=train_labels[6000:]

wandb.run.name="hl_"+str(args.num_layers)+"_nn_"+str(args.hidden_size)+"_lr_"+str(args.learning_rate)+"_ep_"+str(args.epochs)+"_opt_"+args.optimizer+"_bs_"+str(args.batch_size)+"_act_"+args.activation+"_b_"+str(args.beta)+"_init_"+args.weight_init+"_l2_"+str(args.weight_decay)

nn=NeuralNetwork(args.num_layers,
                    args.hidden_size,
                    args.learning_rate,
                    args.epochs,
                    args.batch_size,
                    args.activation,
                    args.beta,
                    args.beta2,
                    args.optimizer,
                    args.weight_init,
                    args.weight_decay,
                    args.loss)
nn.run(train_images,train_labels,validation_images,validation_labels)
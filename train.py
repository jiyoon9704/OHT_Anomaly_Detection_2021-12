import time
import copy
import argparse
from utils import load_train_data
from models import FCAE, RAE, CAE, VAE, CVAE, CRAE, CondAE, ARAE
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import matplotlib as mpl

parser = argparse.ArgumentParser()
# model establishment
parser.add_argument('--model', type=str, default='ARAE', help="models used")
parser.add_argument('--RNN_type', type=str, default='LSTM', help='RNN cell type') # RNN, LSTM, GRU
parser.add_argument('--hidden1', type=int, default=256, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=128, help='Number of units in hidden layer 2.')
parser.add_argument('--latent_dim', type=int, default=10, help='Number of units in latent layer of (C)VAE')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--mid_act',  default=nn.ReLU(), help='Activation function middle layers') # nn.ReLU(), nn.Tanh()
parser.add_argument('--final_act',  default=False, help='Presence of activation for a final layer')
# preprocess hyperparameters
parser.add_argument('--train_date', type=str, default="21-02-05(3)", help='date for training data')
parser.add_argument('--batch_size', type=int, default=64, help='Size of one batch')
parser.add_argument('--time_window', type=int, default=100, help='Length of sliding window')
parser.add_argument('--sliding_step', type=int, default=10, help='Stride of sliding window')
parser.add_argument('--n_features', type=int, default=12, help='Number of sensor channels')
parser.add_argument('--n_labels', type=int, default=9, help='Number of state labels')
# train hyperparameters
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--loss_func',  default=nn.MSELoss(), help='Loss function')
parser.add_argument('--device',  default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='Device')



args = parser.parse_args(args=[])
train_normal = load_train_data(args.train_date, args.batch_size, args.time_window, args.sliding_step, args.model)
input_size = args.time_window * args.n_features #1200
device = args.device

if "RAE" in args.model and args.RNN_type == "RNN":
    args.lr = 0.0005
    args.epochs = 150
elif "RAE" in args.model:
    args.lr = 0.001
    
model_dict = {
            "FCAE": 
             FCAE(input_size, args.hidden1, args.hidden2, args.mid_act, args.final_act),
            "CondAE": 
             CondAE(input_size, args.hidden1, args.hidden2, args.n_labels, args.mid_act, args.final_act),
            "CAE":
             CAE(args.time_window, args.n_features, int(args.hidden1/2), int(args.hidden2/2), args.mid_act, args.final_act),
            "RAE":
             RAE(args.time_window, args.n_features, args.hidden1, RNN_type ="LSTM", final_layer = args.final_act),
            "VAE":
             VAE(input_size,args.hidden1,args.hidden2, args.mid_act, args.final_act, args.latent_dim),
            "CVAE":
             CVAE(input_size,args.hidden1,args.hidden2, args.n_labels, mid_act=args.mid_act,final_layer=args.final_act, latent_size=args.latent_dim),
            "CRAE":
             CRAE(args.time_window, args.n_features, args.hidden1, args.n_labels, RNN_type ="LSTM", final_layer = args.final_act),
            "ARAE":
             ARAE(args.time_window, args.n_features, args.hidden1, args.n_labels, RNN_type ="LSTM", final_layer = args.final_act)
             }

if args.model in model_dict.keys():
    model = model_dict[args.model].to(device)
else:
    raise NotImplementedError
print(model)    
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    
print(device)

def train_model(args, model):
    
    since = time.time()
    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = 100000000

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()            # Set model to training mode
            else:
                model.eval()            # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for (inputs, (labels, index, _, _))in train_normal[phase]:
                inputs = inputs.to(device)                                       
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    if args.model in ["VAE", "CVAE"]:
                        outputs, mu, log_var, _ = model(inputs, labels)
                        loss = args.loss_func(inputs, outputs)
                    elif args.model == "ARAE":
                        outputs, _ = model(inputs, labels)
                        concat_inputs =  torch.cat((inputs, labels), dim=2) #only for context input RAE
                        loss = args.loss_func(concat_inputs, outputs)
                    else:
                        outputs, _ = model(inputs, labels)
                        loss = args.loss_func(outputs, inputs)          


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()                             # perform back-propagation from the loss
                        optimizer.step()                             # perform gradient descent with given optimizer

                # statistics
                running_loss += loss.item() * inputs.size(0)                    

            epoch_loss = running_loss / len(train_normal[phase].dataset)

            print('{} Loss: {:.6f}'.format(phase, epoch_loss))
            
            # deep copy the model
            if phase == 'train':
                train_loss_history.append(epoch_loss)

            if phase == 'val':
                val_loss_history.append(epoch_loss)

            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:6f}'.format(best_val_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history, time_elapsed


# if __name__ == '__main__':
#     best_model, train_loss_history, val_loss_history, time_elapsed = train_model(args, model)
#     print("Validation loss: {:5f}, Elapsed time : {:.0f}m {:.0f}s".format(min(val_loss_history), time_elapsed // 60, time_elapsed % 60))
#     plt.plot(train_loss_history, label='train')
#     plt.plot(val_loss_history, label='val')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend()
#     plt.show()
    
# torch.save(best_model.state_dict(), "best_{}.pt".format(args.model))

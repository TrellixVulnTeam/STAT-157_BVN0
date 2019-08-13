from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn, utils

def net_linear():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net

def net_mlp(hidden_dim1=20, hidden_dim2=10, out_dim=1):
    net = nn.Sequential()
    net.add(nn.Dense(hidden_dim1, activation='relu'))
    #net.add(nn.Dense(hidden_dim2, activation='relu'))
    net.add(nn.Dense(out_dim))
    net.initialize()
    return net
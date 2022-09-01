import os
import numpy as np
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras import regularizers

#import networks.compnet as exp
import networks.tiny as exp
#import networks.mnist as exp

import utils
import my_config
device = my_config.device

x0 = exp.x0
net = exp.net()
net = net.to(device)
net = net.eval()
y0 = net(x0)
y0_np = y0.detach().cpu().numpy()
layers = net.layers
n_layers = len(layers)

# get all outputs
X = []
X.append(x0)
for i,layer in enumerate(layers):
    x_out = layer(X[-1])
    X.append(x_out)

# tensorflow
x0_tf = tf.convert_to_tensor(x0.detach().cpu().numpy())
x0_tf = tf.expand_dims(x0_tf, 0)
tf_net = Sequential()


l2_reg = 0.0
for i,layer in enumerate(layers):
    if (i+1<n_layers) and isinstance(layer, nn.Linear) and isinstance(layers[i+1], nn.ReLU):
        n_input = layer.in_features
        n_output = layer.out_features
        if i==0:
            tf_layer = Dense(n_output, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(l2_reg), input_shape=x0.shape)
        else:
            tf_layer = Dense(n_output, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(l2_reg))
        tf_net.add(tf_layer)
        w_and_b = [layer.weight.detach().cpu().numpy().T,
                   layer.bias.detach().cpu().numpy()]
        tf_net.layers[-1].set_weights(w_and_b)
        print('fully-connected')

    elif isinstance(layer, nn.Linear):
        n_input = layer.in_features
        n_output = layer.out_features
        tf_layer = Dense(n_output, activation=None, kernel_initializer='he_normal')
        tf_net.add(tf_layer)
        w_and_b = [layer.weight.detach().cpu().numpy().T,
                   layer.bias.detach().cpu().numpy()]
        tf_net.layers[-1].set_weights(w_and_b)
        print('fully-connected, no ReLU')

    elif (i+1<n_layers) and isinstance(layer, nn.Conv2d) and isinstance(layers[i+1], nn.ReLU):
        W = utils.conv_matrix(layer,X[i].shape)
        b = utils.conv_bias_vector(layer,X[i+1].shape)
        n_output = W.shape[0]
        if i==0:
            x0_shape = [1,x0.numel()]
            tf_layer = Dense(n_output, activation='relu', kernel_initializer='he_normal', input_shape=x0_shape)
        else:
            tf_layer = Dense(n_output, activation='relu', kernel_initializer='he_normal')
        tf_net.add(tf_layer)
        w_and_b = [W.detach().cpu().numpy().T, b.detach().cpu().numpy()]
        tf_net.layers[-1].set_weights(w_and_b)
        print('Conv2d')

    elif isinstance(layer, nn.Conv2d):
        print('sole Conv2d - NOT IMPLEMENTED!!!')

    elif isinstance(layer, nn.ReLU) and isinstance(layers[i-1], nn.Conv2d):
        print('ReLU')

    elif isinstance(layer, nn.ReLU) and isinstance(layers[i-1], nn.Linear):
        print('ReLU')

    else:
        print('CONVERSION LOOP NOT IMPLEMENTED FOR THIS LAYER!!!')

y0_tf = tf_net(x0_tf)
y0_tf_np = y0_tf.numpy()
print('y0', y0)
print('y0 tf', y0_tf)
print('y0 error', np.linalg.norm(y0_np - y0_tf_np))

save_file = os.path.join(exp.main_dir, 'tf_model.h5')
tf_net.compile()
#tf_net.save_weights(save_file)
tf_net.save(save_file)
#tf_net.save(save_file)

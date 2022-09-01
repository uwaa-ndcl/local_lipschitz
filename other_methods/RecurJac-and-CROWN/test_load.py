import tensorflow as tf
from tensorflow.keras.models import load_model

class NLayerModel:
    def __init__(self, params, restore = None, session=None, use_softmax=False, image_size=28, image_channel=1, activation='relu', activation_param = 0.3, l2_reg = 0.0, dropout_rate = 0.0):
        
        global Sequential, Dense, Dropout, Activation, Flatten, Lambda, Conv2D, MaxPooling2D, LeakyReLU, regularizers, K
        if 'Sequential' not in globals():
            print('importing Keras from tensorflow...')
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
            from tensorflow.keras.layers import Conv2D, MaxPooling2D
            from tensorflow.keras.layers import LeakyReLU
            from tensorflow.keras.models import load_model
            from tensorflow.keras import regularizers
            from tensorflow.keras import backend as K
        
        #self.image_size = image_size
        #self.num_channels = image_channel
        #self.num_labels = 10
        ''' 
        model = Sequential()
        #model.add(Flatten(input_shape=(image_size, image_size, image_channel)))
        # list of all hidden units weights
        self.U = []
        n = 0
        for param in params:
            n += 1
            # add each dense layer, and save a reference to list U
            self.U.append(Dense(param, kernel_initializer = 'he_uniform', kernel_regularizer=regularizers.l2(l2_reg)))
            model.add(self.U[-1])
            # ReLU activation
            # model.add(Activation(activation))
            if activation == "arctan":
                model.add(Lambda(lambda x: tf.atan(x), name=activation+"_"+str(n)))
            elif activation == "leaky":
                print("Leaky ReLU slope: {:.3f}".format(activation_param))
                model.add(LeakyReLU(alpha = activation_param, name=activation+"_"+str(n)))
            else:
                model.add(Activation(activation, name=activation+"_"+str(n)))
            if dropout_rate > 0.0:
                model.add(Dropout(dropout_rate))
        self.W = Dense(10, kernel_initializer = 'he_uniform', kernel_regularizer=regularizers.l2(l2_reg))
        model.add(self.W)
        # output log probability, used for black-box attack
        if use_softmax:
            model.add(Activation('softmax'))
        if restore:
            model.load_weights(restore)
        '''

        modelfile = '/home/trevor/RecurJac-and-CROWN/data/compnet/tf_model.h5'
        model = load_model(modelfile)

        layer_outputs = []
        # save the output of intermediate layers
        for layer in model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer_outputs.append(K.function([model.layers[0].input], [layer.output]))

        # a tensor to get gradients
        self.gradients = []
        for i in range(model.output.shape[1]):
            output_tensor = model.output[:,i]
            self.gradients.append(K.gradients(output_tensor, model.input)[0])

        self.layer_outputs = layer_outputs
        self.model = model
        model.summary()

    def predict(self, data):
        return self.model(data)
    
    def get_gradient(self, data, sess = None):
        if sess is None:
            sess = K.get_session()
        # initialize all un initialized variables
        # sess.run(tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in set(sess.run(tf.report_uninitialized_variables()))]))
        evaluated_gradients = []
        for g in self.gradients:
            evaluated_gradients.append(sess.run(g, feed_dict={self.model.input:data}))
        return evaluated_gradients


modelfile = '/home/trevor/RecurJac-and-CROWN/data/compnet/tf_model.h5'
net = load_model(modelfile)
import pdb; pdb.set_trace()
weight_dims = [7, 20, 30, 10]
activation_param = None
activation = 'relu'

config = tf.compat.v1.ConfigProto() #TDA
config.gpu_options.allow_growth = True
with tf.compat.v1.Session(config=config) as sess: #TDA
    net2 = NLayerModel(weight_dims[:-1], modelfile, activation=activation, activation_param=activation_param)
import pdb; pdb.set_trace()

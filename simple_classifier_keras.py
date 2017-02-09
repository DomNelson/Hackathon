from keras.models import Sequential, Model
from keras.layers import Dense, Activation, merge, Input, Lambda
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn
import sys, os

def generate_data(n, p, r):
    ## Set up the properties of the difference classes. In this case
    ## they will only have different means.
    classes = np.random.randint(0, 10, size=(r, p))

    ## Draw the data
    data = np.zeros((n, p))
    labels = np.zeros((n, r))
    for i in range(n):
        c = np.random.choice(range(r))
        data[i] = np.random.multivariate_normal(classes[c], np.identity(p))

        ## Labels are given by a '1' in the corresponding
        ## class column
        labels[i, c] = 1.

    return data, labels

layers_list = [1, 2, 3]
nodes = 5
repeats = 20
epochs = 50
loss_hists = np.zeros((repeats, epochs))
mean_loss = np.zeros((len(layers_list), epochs))
for j, layers in enumerate(layers_list):
    for r in range(repeats):
        ## Draw data
        n, p = 100, 20
        r = 2
        data, labels = generate_data(n, p, r)

        ## Normalize the data so that every value is between 0 and 1
        data = data / np.max(data)

        ## Layer which takes data as input
        data_input = Input(shape=(data.shape[1],), name='input')
        last_layer = data_input

        ## Now add multiple hidden layers
        for l in range(layers):
            data_layer = Dense(nodes, activation='relu')(last_layer)
            last_layer = data_layer

        ## Finally, the output layer gives a prediction
        output_layer = Dense(r, name='output', activation='sigmoid')(data_layer)

        ## Now create a model from the layers we have constructed
        model = Model(input=data_input, output=output_layer)

        model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['categorical_accuracy'])

        history = model.fit(data, labels, nb_epoch=epochs, batch_size=1, validation_split=0.3, verbose=1)
        loss_hists[r, :] = history.history['val_loss']

    mean_loss[j, :] = np.average(loss_hists, axis=0)

## Plot the validation losses over the parameters provided
for i, num_layers in enumerate(layers_list):
    plt.plot(mean_loss[i], label=str(num_layers) + ' layers')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss (average over 20 training sessions)')
plt.legend()
plt.title('Validation loss as a function of the number of hidden layers, \n' +\
            'where each layer contains 5 nodes')
plt.savefig(os.path.expanduser('~/Documents/Courses/MATH680/project/' +\
            'resources/simple_loss_5layers.png'))

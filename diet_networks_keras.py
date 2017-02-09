from keras.models import Sequential, Model
from keras.layers import Dense, Activation, merge, Input
import numpy as np
from keras.utils.visualize_util import plot

## Data obtained from:
## http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/
nodes = 100
datafile = 'ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.vcf'
data = np.genfromtxt(datafile, dtype=str, usecols=range(10, 2300), max_rows=2000)

## Clean rows with missing data, denoted by "./."
data = np.asarray([d for d in data if "./." not in d])

## Convert to unphased data - only a 0, 1, 2 for each individual and normalize
unphase = np.vectorize(lambda x: np.sum(map(int, x.split('/'))))
data = unphase(data)
data = data / np.max(data)


def select_weights(tensor_list):
    ## Return the last tensor, ignoring preceding ones
    return tensor_list[-1]


def square_data(data):
    ## Tiles data until it is square - this way
    ## we can separately provide the (non-square) transpose
    ## and still have the same number of samples
    while data.shape[0] < data.shape[1]:
        data = np.vstack([data, data])

    return data[:data.shape[1]]

## Layer which takes as input the data in its original form
data_input = Input(shape=(data.shape[1],), name='data input')

## Layers which learns an embedding from the transposed data
data_transpose_input = Input(shape=(data.shape[0],), name='transposed data input')
transpose_layer = Dense(nodes, activation='sigmoid')(data_transpose_input)
transpose_layer = Dense(data.shape[1], activation='sigmoid')(transpose_layer)

## Merge the layers, but take only the weights from the transposed data
merged = merge([data_input, transpose_layer], mode=select_weights,
                        output_shape=(nodes,))
merged = Dense(nodes, activation='sigmoid')(merged)
merged = Dense(data.shape[1], activation='sigmoid')(merged)

## Layers which attempt to reconstruct the original data from the embedding
reconstruction_input = Input(shape=(data.shape[0],), name='reconstruction input')
reconstruction_weights = Dense(nodes, activation='sigmoid')(reconstruction_input)
reconstruction_weights = Dense(data.shape[1],
                            activation='sigmoid')(reconstruction_weights)

## Now perform the reconstruction with the reconstruction weights
reconstruction = merge([merged, reconstruction_weights], mode=select_weights,
                    output_shape=(data.shape[1],), name='reconstruction')

## Now create a model from the layers we have constructed
model = Model(input=[data_input, data_transpose_input, reconstruction_input],
             output=[reconstruction])


plot(model, to_file='full_model.png', show_shapes=True)

model.compile(optimizer='Adagrad', loss='binary_crossentropy')

model.fit([square_data(data), data.transpose(), data.transpose()],
        [square_data(data)], nb_epoch=50, batch_size=32)

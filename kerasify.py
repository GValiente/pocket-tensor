import numpy as np
import struct

LAYER_DENSE = 1
LAYER_CONV_1D = 2
LAYER_CONV_2D = 3
LAYER_LOCALLY_1D = 4
LAYER_FLATTEN = 6
LAYER_ELU = 7
LAYER_ACTIVATION = 8
LAYER_MAXPOOLING_2D = 9
LAYER_LSTM = 10
LAYER_EMBEDDING = 11
LAYER_BATCH_NORMALIZATION = 12
LAYER_LEAKY_RELU = 13

ACTIVATION_LINEAR = 1
ACTIVATION_RELU = 2
ACTIVATION_ELU = 3
ACTIVATION_SOFTPLUS = 4
ACTIVATION_SOFTSIGN = 5
ACTIVATION_SIGMOID = 6
ACTIVATION_TANH = 7
ACTIVATION_HARD_SIGMOID = 8
ACTIVATION_SOFTMAX = 9
ACTIVATION_SELU = 10


def write_tensor(f, data, dims=1):
    '''
    Writes tensor as flat array of floats to file in 1024 chunks,
    prevents memory explosion writing very large arrays to disk
    when calling struct.pack().
    '''
    for stride in data.shape[:dims]:
        f.write(struct.pack('I', stride))

    data = data.flatten()
    step = 1024
    written = 0

    for i in np.arange(0, len(data), step):
        remaining = min(len(data) - i, step)
        written += remaining
        f.write(struct.pack('=%sf' % remaining, *data[i: i + remaining]))

    assert written == len(data)


def export_activation(f, activation):
    if activation == 'linear':
        f.write(struct.pack('I', ACTIVATION_LINEAR))
    elif activation == 'relu':
        f.write(struct.pack('I', ACTIVATION_RELU))
    elif activation == 'elu':
        f.write(struct.pack('I', ACTIVATION_ELU))
    elif activation == 'softplus':
        f.write(struct.pack('I', ACTIVATION_SOFTPLUS))
    elif activation == 'softsign':
        f.write(struct.pack('I', ACTIVATION_SOFTSIGN))
    elif activation == 'sigmoid':
        f.write(struct.pack('I', ACTIVATION_SIGMOID))
    elif activation == 'tanh':
        f.write(struct.pack('I', ACTIVATION_TANH))
    elif activation == 'hard_sigmoid':
        f.write(struct.pack('I', ACTIVATION_HARD_SIGMOID))
    elif activation == 'softmax':
        f.write(struct.pack('I', ACTIVATION_SOFTMAX))
    elif activation == 'selu':
        f.write(struct.pack('I', ACTIVATION_SELU))
    else:
        assert False, "Unsupported activation type: %s" % activation


def export_layer_normalization(f, layer):
    epsilon = layer.epsilon
    gamma = layer.get_weights()[0]
    beta = layer.get_weights()[1]
    pop_mean = layer.get_weights()[2]
    pop_variance = layer.get_weights()[3]

    weights = gamma / np.sqrt(pop_variance + epsilon)
    biases = beta - pop_mean * weights

    f.write(struct.pack('I', LAYER_BATCH_NORMALIZATION))

    write_tensor(f, weights)
    write_tensor(f, biases)


def export_layer_dense(f, layer):
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    activation = layer.get_config()['activation']

    weights = weights.transpose()
    # shape: (outputs, dims)

    f.write(struct.pack('I', LAYER_DENSE))

    write_tensor(f, weights, 2)
    write_tensor(f, biases)

    export_activation(f, activation)


def export_layer_conv1d(f, layer):
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    activation = layer.get_config()['activation']

    weights = weights.transpose(2, 0, 1)
    # shape: (outputs, steps, dims)

    f.write(struct.pack('I', LAYER_CONV_1D))
    write_tensor(f, weights, 3)
    write_tensor(f, biases)
    export_activation(f, activation)


def export_layer_conv2d(f, layer):
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    activation = layer.get_config()['activation']

    weights = weights.transpose(3, 0, 1, 2)
    # shape: (outputs, rows, cols, depth)

    f.write(struct.pack('I', LAYER_CONV_2D))
    write_tensor(f, weights, 4)
    write_tensor(f, biases)

    export_activation(f, activation)


def export_layer_locally1d(f, layer):
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    activation = layer.get_config()['activation']

    weights = weights.transpose(0, 2, 1)
    # shape: (new_steps, outputs, ksize*dims)

    f.write(struct.pack('I', LAYER_LOCALLY_1D))

    write_tensor(f, weights, 3)
    write_tensor(f, biases, 2)

    export_activation(f, activation)


def export_layer_maxpooling2d(f, layer):
    pool_size = layer.get_config()['pool_size']

    f.write(struct.pack('I', LAYER_MAXPOOLING_2D))
    f.write(struct.pack('I', pool_size[0]))
    f.write(struct.pack('I', pool_size[1]))


def export_layer_lstm(f, layer):
    inner_activation = layer.get_config()['recurrent_activation']
    activation = layer.get_config()['activation']
    return_sequences = int(layer.get_config()['return_sequences'])

    weights = layer.get_weights()
    units = layer.units

    W_i = weights[0][:, :units].transpose()
    W_f = weights[0][:, units: units*2].transpose()
    W_c = weights[0][:, units*2: -units].transpose()
    W_o = weights[0][:, -units:].transpose()

    U_i = weights[1][:, :units].transpose()
    U_f = weights[1][:, units: units*2].transpose()
    U_c = weights[1][:, units*2: -units].transpose()
    U_o = weights[1][:, -units:].transpose()

    b_i = weights[2][:units].reshape((1, -1))
    b_f = weights[2][units: units*2].reshape((1, -1))
    b_c = weights[2][units*2: -units].reshape((1, -1))
    b_o = weights[2][-units:].reshape((1, -1))

    f.write(struct.pack('I', LAYER_LSTM))

    write_tensor(f, W_i, 2)
    write_tensor(f, U_i, 2)
    write_tensor(f, b_i, 2)

    write_tensor(f, W_f, 2)
    write_tensor(f, U_f, 2)
    write_tensor(f, b_f, 2)

    write_tensor(f, W_c, 2)
    write_tensor(f, U_c, 2)
    write_tensor(f, b_c, 2)

    write_tensor(f, W_o, 2)
    write_tensor(f, U_o, 2)
    write_tensor(f, b_o, 2)

    export_activation(f, inner_activation)
    export_activation(f, activation)
    f.write(struct.pack('I', return_sequences))


def export_layer_embedding(f, layer):
    weights = layer.get_weights()[0]

    f.write(struct.pack('I', LAYER_EMBEDDING))
    write_tensor(f, weights, 2)


def export_model(model, filename):
    with open(filename, 'wb') as f:
        model_layers = [
            l for l in model.layers if type(l).__name__ not in ['Dropout']]
        num_layers = len(model_layers)
        f.write(struct.pack('I', num_layers))

        for layer in model_layers:
            layer_type = type(layer).__name__

            if layer_type == 'Dense':
                export_layer_dense(f, layer)

            elif layer_type == 'Conv1D':
                export_layer_conv1d(f, layer)

            elif layer_type == 'Conv1D':
                export_layer_conv1d(f, layer)

            elif layer_type == 'Conv2D':
                export_layer_conv2d(f, layer)

            elif layer_type == 'LocallyConnected1D':
		export_layer_locally1d(f, layer)

            elif layer_type == 'Flatten':
                f.write(struct.pack('I', LAYER_FLATTEN))

            elif layer_type == 'ELU':
                f.write(struct.pack('I', LAYER_ELU))
                f.write(struct.pack('f', layer.alpha))

            elif layer_type == 'Activation':
                activation = layer.get_config()['activation']
                f.write(struct.pack('I', LAYER_ACTIVATION))
                export_activation(f, activation)

            elif layer_type == 'MaxPooling2D':
                export_layer_maxpooling2d(f, layer)

            elif layer_type == 'LSTM':
                export_layer_lstm(f, layer)

            elif layer_type == 'Embedding':
                export_layer_embedding(f, layer)

            elif layer_type == 'BatchNormalization':
                export_layer_normalization(f, layer)

            elif layer_type == 'LeakyReLU':
                f.write(struct.pack('I', LAYER_LEAKY_RELU))
                f.write(struct.pack('f', layer.alpha))

            else:
                assert False, "Unsupported layer type: %s" % layer_type

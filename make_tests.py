import tensorflow

import numpy as np
import os
import pprint
import re
import errno
import sys

try:
    from keras import backend as K
    from keras.models import Sequential
    from keras.layers import (
        Conv1D, Conv2D, LocallyConnected1D, Dense, Flatten, Activation,
        MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, RepeatVector
    )
    from keras.layers.recurrent import LSTM
    from keras.layers.advanced_activations import ELU, LeakyReLU
    from keras.layers.embeddings import Embedding
    from keras.engine.input_layer import Input
    from keras.models import Model
except:
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv1D, Conv2D, LocallyConnected1D, Dense, Flatten, Activation,
        MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, RepeatVector
    )
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import ELU, LeakyReLU
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.engine.input_layer import Input
    from tensorflow.keras.models import Model

from tensorflow import ConfigProto, Session
from pt import export_model

# Fix random seed:
np.random.seed(1)
tensorflow.set_random_seed(2)

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
try:
    K.tensorflow_backend.set_session(Session(config=config))
except:
    K.set_session(Session(config=config))

np.set_printoptions(precision=25, threshold=sys.maxsize)

src_path = 'tests/src'
models_path = 'tests/models'

try:
    os.makedirs(src_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(models_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


def c_array(a):
    s = pprint.pformat(a.flatten())

    s = re.sub(r'[ \t\n]*', '', s)
    s = re.sub(r'[ \t]*,[ \t]*', ', ', s)
    s = re.sub(r'[ \t]*\][, \t]*', '} ', s)
    s = re.sub(r'[ \t]*\[[ \t]*', '{', s)
    s = s.replace('array(', '').replace(')', '')
    s = re.sub(r'[, \t]*dtype=float32', '', s)
    s = s.strip()

    if a.shape:
        shape = repr(a.shape)
        shape = re.sub(r',*\)', '}', shape.replace('(', '{'))
    else:
        shape = '{1}'
    return shape, s


TEST_CASE = '''/* Autogenerated file, DO NOT EDIT */
#include "test_util.h"

TEST_CASE("%s")
{
    pt::Tensor in%s;
    in.setData(%s);

    pt::Tensor expected%s;
    expected.setData(%s);

    testModel(in, expected, "%s", %sf);
}
'''


def output_testcase(model, test_x, test_y, name, eps):
    print('Processing %s' % name)
    model.compile(loss='mse', optimizer='adam')
    model.fit(test_x, test_y, epochs=1, verbose=False)
    predict_y = model.predict(test_x).astype('f')
    print(model.summary())

    export_model(model, models_path + '/%s.model' % name)

    with open(src_path + '/%s_test.cpp' % name, 'w') as f:
        x_shape, x_data = c_array(test_x[0])
        y_shape, y_data = c_array(predict_y[0])

        f.write(TEST_CASE % (
            name, x_shape, x_data, y_shape, y_data, name, eps))


''' Dense 1x1 '''
test_x = np.arange(10)
test_y = test_x * 10 + 1
model = Sequential([
    Dense(1, input_dim=1)
])
output_testcase(model, test_x, test_y, 'dense_1x1', '1e-6')


''' Dense 10x1 '''
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential([
    Dense(1, input_dim=10)
])
output_testcase(model, test_x, test_y, 'dense_10x1', '1e-6')


''' Dense 2x2 '''
test_x = np.random.rand(10, 2).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential([
    Dense(2, input_dim=2),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'dense_2x2', '1e-6')


''' Dense 10x10 '''
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential([
    Dense(10, input_dim=10),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'dense_10x10', '1e-6')


''' Dense 10x10x10 '''
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10, 10).astype('f')
model = Sequential([
    Dense(10, input_dim=10),
    Dense(10)
])
output_testcase(model, test_x, test_y, 'dense_10x10x10', '1e-6')


''' Conv1D 2 '''
test_x = np.random.rand(10, 2, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv1D(1, 2, input_shape=(2, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv1d_2', '1e-6')


''' Conv1D 3 '''
test_x = np.random.rand(10, 3, 1).astype('f').astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv1D(1, 3, input_shape=(3, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv1d_3', '1e-6')


''' Conv1D3x3 '''
test_x = np.random.rand(10, 10, 3).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv1D(3, 3, input_shape=(10, 3)),
    Flatten(),
    BatchNormalization(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv1d_3x3', '1e-6')


''' Conv 2x2 '''
test_x = np.random.rand(10, 2, 2, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(1, (2, 2), input_shape=(2, 2, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_2x2', '1e-6')


''' Conv 3x3 '''
test_x = np.random.rand(10, 3, 3, 1).astype('f').astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(1, (3, 3), input_shape=(3, 3, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_3x3', '1e-6')


''' Conv 3x3x3 '''
test_x = np.random.rand(10, 10, 10, 3).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(3, (3, 3), input_shape=(10, 10, 3)),
    Flatten(),
    BatchNormalization(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_3x3x3', '1e-6')


''' LocallyConnected1D 2 '''
test_x = np.random.rand(10, 2, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    LocallyConnected1D(1, 2, input_shape=(2, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'locally_connected_1d_2', '1e-6')


''' LocallyConnected1D 3 '''
test_x = np.random.rand(10, 3, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    LocallyConnected1D(1, 3, input_shape=(3, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'locally_connected_1d_3', '1e-6')


''' LocallyConnected1D 3x3 '''
test_x = np.random.rand(10, 10, 3).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    LocallyConnected1D(3, 3, input_shape=(10, 3)),
    Flatten(),
    BatchNormalization(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'locally_connected_1d_3x3', '1e-6')


''' Activation ELU '''
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 1).astype('f')
model = Sequential([
    Dense(10, input_dim=10),
    ELU(alpha=0.5),
    Dense(1, activation='elu')
])
output_testcase(model, test_x, test_y, 'elu_10', '1e-6')


''' Activation relu '''
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential([
    Dense(10, input_dim=10),
    Activation('relu')
])
output_testcase(model, test_x, test_y, 'relu_10', '1e-6')


''' Activation LeakyReLU '''
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 1).astype('f')
model = Sequential([
    Dense(10, input_dim=10),
    LeakyReLU(alpha=0.5),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'leaky_relu_10', '1e-6')


''' Dense relu '''
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential([
    Dense(10, input_dim=10, activation='relu'),
    Dense(10, input_dim=10, activation='relu'),
    Dense(10, input_dim=10, activation='relu')
])
output_testcase(model, test_x, test_y, 'dense_relu_10', '1e-6')


''' Dense elu '''
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential([
    Dense(10, input_dim=10, activation='elu'),
    Dense(10, input_dim=10, activation='elu'),
    Dense(10, input_dim=10, activation='elu')
])
output_testcase(model, test_x, test_y, 'dense_elu_10', '1e-6')


''' Dense softsign '''
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential([
    Dense(10, input_dim=10, activation='softsign'),
    Dense(10, input_dim=10, activation='softsign'),
    Dense(10, input_dim=10, activation='softsign')
])
output_testcase(model, test_x, test_y, 'dense_softsign_10', '1e-6')


''' Dense softmax '''
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential([
    Dense(10, input_dim=10, activation='softmax'),
    Dense(10, input_dim=10, activation='softmax'),
    Dense(10, input_dim=10, activation='softmax')
])
output_testcase(model, test_x, test_y, 'dense_softmax_10', '0.05')


''' Dense tanh '''
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential([
    Dense(10, input_dim=10, activation='tanh'),
    Dense(10, input_dim=10, activation='tanh'),
    Dense(10, input_dim=10, activation='tanh')
])
output_testcase(model, test_x, test_y, 'dense_tanh_10', '1e-6')


''' Dense selu '''
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential([
    Dense(10, input_dim=10, activation='selu'),
    Dense(10, input_dim=10, activation='selu'),
    Dense(10, input_dim=10, activation='selu')
])
output_testcase(model, test_x, test_y, 'dense_selu_10', '1e-6')


''' Conv softplus '''
test_x = np.random.rand(10, 2, 2, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(1, (2, 2), input_shape=(2, 2, 1), activation='softplus'),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_softplus_2x2', '1e-6')


''' Conv hardsigmoid '''
test_x = np.random.rand(10, 2, 2, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(1, (2, 2), input_shape=(2, 2, 1), activation='hard_sigmoid'),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_hard_sigmoid_2x2', '1e-6')


''' Conv sigmoid '''
test_x = np.random.rand(10, 2, 2, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(1, (2, 2), input_shape=(2, 2, 1), activation='sigmoid'),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_sigmoid_2x2', '1e-6')


''' Maxpooling2D 1x1'''
test_x = np.random.rand(10, 10, 10, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    MaxPooling2D(pool_size=(1, 1), input_shape=(10, 10, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'maxpool2d_1x1', '1e-6')


''' Maxpooling2D 2x2'''
test_x = np.random.rand(10, 10, 10, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    MaxPooling2D(pool_size=(2, 2), input_shape=(10, 10, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'maxpool2d_2x2', '1e-6')


''' Maxpooling2D 3x2x2'''
test_x = np.random.rand(10, 10, 10, 3).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    MaxPooling2D(pool_size=(2, 2), input_shape=(10, 10, 3)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'maxpool2d_3x2x2', '1e-6')


''' Maxpooling2D 3x3x3'''
test_x = np.random.rand(10, 10, 10, 3).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    MaxPooling2D(pool_size=(3, 3), input_shape=(10, 10, 3)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'maxpool2d_3x3x3', '1e-6')


''' Maxpooling2D 8x2x2'''
test_x = np.random.rand(10, 10, 10, 8).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    MaxPooling2D(pool_size=(2, 2), input_shape=(10, 10, 8)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'maxpool2d_8x2x2', '1e-6')


''' Maxpooling2D 8x3x3'''
test_x = np.random.rand(10, 10, 10, 8).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    MaxPooling2D(pool_size=(3, 3), input_shape=(10, 10, 8)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'maxpool2d_8x3x3', '1e-6')


''' GlobalMaxpooling2D 1'''
test_x = np.random.rand(10, 10, 10, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    GlobalMaxPooling2D(input_shape=(10, 10, 1))
])
output_testcase(model, test_x, test_y, 'global_maxpool2d_1', '1e-6')


''' GlobalMaxpooling2D 3'''
test_x = np.random.rand(10, 10, 10, 3).astype('f')
test_y = np.random.rand(10, 3).astype('f')
model = Sequential([
    GlobalMaxPooling2D(input_shape=(10, 10, 3))
])
output_testcase(model, test_x, test_y, 'global_maxpool2d_3', '1e-6')


''' GlobalMaxpooling2D 8'''
test_x = np.random.rand(10, 10, 10, 8).astype('f')
test_y = np.random.rand(10, 8).astype('f')
model = Sequential([
    GlobalMaxPooling2D(input_shape=(10, 10, 8))
])
output_testcase(model, test_x, test_y, 'global_maxpool2d_8', '1e-6')


''' LSTM simple 7x20 '''
test_x = np.random.rand(10, 7, 20).astype('f')
test_y = np.random.rand(10, 3).astype('f')
model = Sequential([
    LSTM(3, return_sequences=False, input_shape=(7, 20))
])
output_testcase(model, test_x, test_y, 'lstm_simple_7x20', '1e-6')


''' LSTM simple stacked 16x9 '''
test_x = np.random.rand(10, 16, 9).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    LSTM(16, return_sequences=False, input_shape=(16, 9)),
    Dense(3, input_dim=16, activation='tanh'),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'lstm_simple_stacked_16x9', '1e-6')


''' LSTM stacked 64x83 '''
test_x = np.random.rand(10, 64, 83).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    LSTM(16, return_sequences=True, input_shape=(64, 83)),
    LSTM(16, return_sequences=False),
    Dense(1, activation='sigmoid')
])
output_testcase(model, test_x, test_y, 'lstm_stacked_64x83', '1e-6')


''' Embedding 64 '''
np.random.seed(10)
test_x = np.random.randint(100, size=(32, 10)).astype('f')
test_y = np.random.rand(32, 20).astype('f')
model = Sequential([
    Embedding(100, 64, input_length=10),
    Flatten(),
    Dense(20, activation='sigmoid')
])
output_testcase(model, test_x, test_y, 'embedding_64', '1e-6')


''' Input '''
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')
a = Input(shape=(10,))
b = Dense(1)(a)
model = Model(inputs=a, outputs=b)
output_testcase(model, test_x, test_y, 'input', '1e-6')


''' RepeatVector '''
test_x = np.random.rand(10, 32).astype('f')
test_y = np.random.rand(10, 3, 32).astype('f')
model = Sequential([
    Dense(32, input_dim=32),
    RepeatVector(3)
])
output_testcase(model, test_x, test_y, 'repeat_vector', '1e-6')


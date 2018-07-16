import sys
import datetime
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb


max_features = 20000
maxlen = 80
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_train = x_train[:1]
y_train = y_train[:1]
x_test = x_test[:1]
y_test = y_test[:1]

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=(x_test, y_test))

print('Measure...')
min_elapsed_mcs = sys.maxint

for i in xrange(0, 9):
    start_time = datetime.datetime.now()
    predict_output = model.predict(x_test)
    elapsed_time = datetime.datetime.now() - start_time
    elapsed_mcs = elapsed_time.microseconds
    min_elapsed_mcs = min(elapsed_mcs, min_elapsed_mcs)

print('Elapsed mcs: ' + str(min_elapsed_mcs))

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def build_model_for_grid(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[11]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model

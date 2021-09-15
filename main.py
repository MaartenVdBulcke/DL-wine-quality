import pandas as pd


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from utils.manipulate_data import *
from utils.visualise import *
from utils.create_model import *
from utils.grid_search import *


wine_full = pd.read_csv('../../additional_resources/datasets/Wine Quality/wine.csv')
wine_full.drop('index', axis=1, inplace=True)
wine_full.drop_duplicates(inplace=True)
wine_full.reset_index(inplace=True, drop=True)
wine_full = wine_full.sample(frac=1).reset_index(drop=True)  # Shuffle dataframe


if __name__ == '__main__':
    wine_binary = wine_full.copy()
    wine_binary = make_target_binary(wine_binary, 'quality')
    X, y = split_features_target(wine_binary, 'quality')
    X_norm = StandardScaler().fit_transform(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X_norm, y, test_size=0.2,
                                                                  random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2,
                                                      random_state=42, stratify=y_train_full)

    # perform_random_grid_and_grid_search(X_train, y_train, X_val, y_val, X_test, y_test)


    # create model based on grid search
    number_of_neurons = 40
    learning_rate = 0.0033

    model = keras.models.Sequential([
        keras.layers.Dense(number_of_neurons, input_shape=(11,), activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    checkpoint_callback = keras.callbacks.ModelCheckpoint('best_onlybest_grid.h5', save_best_only=True,
                                                          monitor='val_accuracy')
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                        # callbacks=[keras.callbacks.EarlyStopping(patience=10)])
                        callbacks=[checkpoint_callback, keras.callbacks.EarlyStopping(patience=30,
                                                                                      monitor='val_accuracy')])
    # model.save('models/model_grid_search.h5')

    model = keras.models.load_model('best_onlybest_grid.h5')

    plot_loss_accuracy(history)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    results = model.evaluate(X_train, y_train, batch_size=32)
    print('train:', results)
    results = model.evaluate(X_val, y_val, batch_size=32)
    print('val:', results)
    results = model.evaluate(X_test, y_test, batch_size=32)
    print(f"Accuracy on test set is {results[1] * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    print(tf.math.confusion_matrix(y_test, y_pred))


# COACHES NOTES: Super slick! I would add more comments through out in the case you share this with others.
# who wouldn't have access to the play notebooks. 
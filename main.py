import pandas as pd

from utils.manipulate_data import *
from utils.visualise import *
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

wine_full = pd.read_csv('../../additional_resources/datasets/Wine Quality/wine.csv')
wine_full.drop('index', axis=1, inplace=True)
wine_full.drop_duplicates(inplace=True)
wine_full.reset_index(inplace=True, drop=True)
wine_full = wine_full.sample(frac=1).reset_index(drop=True)  # Shuffle dataframe

if __name__ == '__main__':
    wine_binary = wine_full.copy()
    wine_binary = make_target_binary(wine_binary, 'quality')
    X_train, X_test, y_train, y_test = split_dataset_in_train_test(df=wine_binary,
                                                                   target='quality',
                                                                   test_size=0.20)

    # create model
    model = keras.models.Sequential([
        keras.layers.Dense(6, input_shape=(11,), activation='relu'),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.save('models/model_v1')


    history = model.fit(X_train, y_train, epochs=100, batch_size=32)

    plot_loss_accuracy(history)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    results = model.evaluate(X_test, y_pred, batch_size=32)
    print(f"Accuracy on test set is {results[1] * 100:.2f}%")

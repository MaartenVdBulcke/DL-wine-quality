
if __name__=='__main__':


    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import classification_report
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


    from create_model import *
    from visualise import *




def perform_random_grid_and_grid_search(X_train, y_train, X_val, y_val, X_test, y_test):
    keras_classifier = keras.wrappers.scikit_learn.KerasClassifier(build_model_for_grid)

    param_distribs_random_grid = {
        'n_hidden': np.arange(1, 6).tolist(),
        'n_neurons': np.arange(1, 100).tolist(),
        'learning_rate': np.arange(0.0001, 0.004, step=0.00005)
    }

    random_search_cv = RandomizedSearchCV(keras_classifier, param_distribs_random_grid, n_iter=25, cv=3)

    random_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                         callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    # print(random_search_cv.best_params_)

    # grid search based on random search
    param_grid = {
        'n_neurons': np.arange(40, 60, step=2).tolist(),
        'learning_rate': np.arange(0.00250, 0.00350, step=0.0001).tolist()
    }

    keras_class_grid = keras.wrappers.scikit_learn.KerasClassifier(build_model_for_grid)
    grid_search_cv = GridSearchCV(keras_class_grid, param_grid, cv=3)
    grid_search_cv.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val),
                       callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    # create model
    model = keras.models.Sequential([
        keras.layers.Dense(100, input_shape=(11,), activation='relu'),
        # keras.layers.Dense(100, activation='sigmoid'),
        # keras.layers.Dense(60, activation='sigmoid'),
        keras.layers.Dense(50, activation='relu'),
        # keras.layers.Dense(20, activation='sigmoid'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # model.save('models/model.h5')


    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=.2)

    plot_loss_accuracy(history)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    results = model.evaluate(X_test, y_test, batch_size=32)
    print(f"Accuracy on test set is {results[1] * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    print(tf.math.confusion_matrix(y_test, y_pred))
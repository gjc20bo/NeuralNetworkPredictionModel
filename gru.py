import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import copy


def launchGRU(data_train, data_test, stockName):
    acc_logs = []
    trim_test = 827
    trim_train = 2000


    class lossCallback(tf.keras.callbacks.Callback):
        def __init__(self, x_test, y_test):
            super().__init__()
            self.x_test = x_test
            self.y_test = y_test

        def on_epoch_end(self, epoch, logs=None):
            acc = self.model.evaluate(self.x_test, self.y_test, batch_size=1, verbose=0)
            acc_logs.append(acc)

    # Normalize inputs. Note: Training data should use fit_transform, while testing data should use transform.
    sc = MinMaxScaler(feature_range=(-1, 1))
    sc_data = sc.fit_transform(data_train.values[trim_train:, 1:])


    x_train = sc_data

    y_train = sc_data[1:]
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = x_train[1:]
    x_train = x_train[:len(x_train)-1]

    x_test = sc.transform(data_test.values[:, 1:])

    y_test = x_test[1:len(x_test)-trim_test]
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_test = x_test[:len(x_test)-trim_test-1]

    es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)


    model = keras.Sequential()
    model.add(layers.GRU(50, return_sequences=False,input_shape=(x_train.shape[1],1)))
    model.add(layers.Dense(4))
    #model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.0001), loss='mse', metrics='RootMeanSquaredError')

    loss_history = model.fit(x_train, y_train, batch_size=16, epochs=40, callbacks=[lossCallback(x_test, y_test)])


    accuracy = {'fit_loss':None, 'evaluation_loss':None}
    accuracy['fit']=loss_history.history['loss']

    accuracy['evaluation']=acc_logs


    plt.figure(figsize=(16,8))
    plt.title('Gru Loss per Epoch for:'+stockName)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.plot([i for i in range(len(acc_logs))], loss_history.history['loss'])
    plt.plot([i for i in range(len(acc_logs))], accuracy['evaluation'])

    plt.legend(['fit_loss', 'evaluation_loss', 'evaluation_rmse'], loc='lower right')


    future_prediction = model.predict(x_test)

    future_prediction = sc.inverse_transform(future_prediction)

    test_data = copy.deepcopy(data_test[1:len(data_test) - trim_test])

    test_data['Close'] = data_test.values[1:len(data_test) - trim_test, 3:]
    test_data['Predictions'] = future_prediction[:, 3:]
    plt.figure(figsize=(16,8))
    plt.title('Gru Model for ' + stockName)
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')

    plt.plot(test_data[['Close', 'Predictions']])

    plt.legend(['Actual', 'Predicted'], loc='lower right')



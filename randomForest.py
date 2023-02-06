import numpy as np
import pandas as pd
import sklearn.ensemble
from sklearn.preprocessing import MinMaxScaler


def randomForest(data_train, data_test, stockName):
    # Normalize inputs. Note: Training data should use fit_transform, while testing data should use transform.
    sc = MinMaxScaler(feature_range=(-1, 1))
    sc_data = sc.fit_transform(data_train.values[1000:,1:])


    x_train = sc_data

    y_train = sc_data[1:]
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = x_train[1:]

    x_train = x_train[:len(x_train)-1]

    x_test = sc.transform(data_test.values[:,1:])

    y_test = x_test[1:len(x_test)-827]
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_test = x_test[:len(x_test)-828]

    y_binary = [0]

    for i in range(0, len(x_train)-1):
        if x_train[i,3] > y_train[i+1,3]:
            y_binary.append(0)
        else:
            y_binary.append(1)
    y_test_binary =[0]
    for j in range(0, len(x_test)-1):
        if x_test[j,3] > y_test[j+1,3]:
            y_test_binary.append(0)
        else:
            y_test_binary.append(1)

    y_binary = np.array(y_binary).reshape(-1,1)
    y_test_binary = np.array(y_test_binary).reshape(-1,1)



    model = sklearn.ensemble.RandomForestClassifier()

    model.fit(x_train, y_binary.ravel())
    other_prediction = []
    predictions = model.predict(x_test)

    for i in range(0,len(y_test_binary)):
        other_prediction.append(y_test_binary[i,0])


    print("The accuracy for ",stockName, " is: ", sklearn.metrics.accuracy_score(other_prediction,predictions))

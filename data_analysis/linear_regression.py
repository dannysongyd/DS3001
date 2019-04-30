import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import pickle

def transform_data (city, *argv):
    x = [0] * 262
    x[200+city_list.index(city)] = 1
    for arg in argv:
        x[attr_list.index(arg)] = 1
    return x

def fMSE (Y, Yhat):
    return np.mean((Y - Yhat)**2, axis=0)

if __name__ == "__main__":
    data = pd.read_csv('final_data.csv')
    cities = pd.read_csv('final_cities.csv')
    data = data.drop(['Unnamed: 0'], axis=1)
    cities = cities.drop(['Unnamed: 0'], axis=1)
    # TODO: this two list is for the transform_data
    attr_list = list(data)[:-2]
    city_list = list(cities)

    X = data.values[:,:-2]
    X = np.hstack((X, cities.values))
    Y = data.values[:,-2:]

    # indices = np.argsort(Y, axis=0)[-5000:][:,0]
    # Y = Y[indices]
    # X = X[indices]

    # split data set
    N = len(X)
    permutations = np.random.permutation(N)
    train_idx = permutations[:(N//2)]
    test_idx = permutations[(N//2):]
    trainX = X[train_idx]
    trainY = Y[train_idx]
    testX = X[test_idx]
    testY = Y[test_idx]


    # NOTE: Not used linear regression
    # reg = LinearRegression().fit(trainX, trainY)
    # print('Training score:', reg.score(trainX, trainY))
    # print('Testing score:', reg.score(testX, testY))
    # print('Training MSE:', np.mean((reg.predict(trainX) - trainY)**2, axis=0))
    # print('Testing MSE:', np.mean((reg.predict(testX) - testY)**2, axis=0))

    # SGD Linear regression
    sgd1 = SGDRegressor(max_iter=1000, tol=1e-3)
    sgd2 = SGDRegressor(max_iter=1000, tol=1e-3)
    sgd1.fit(trainX, trainY[:,0]) # comments
    sgd2.fit(trainX, trainY[:,1]) # stars

    print('SGD1 - comments')
    print('Training score:', sgd1.score(trainX, trainY[:,0]))
    print('Testing score:', sgd1.score(testX, testY[:,0]))
    print('Training MSE:', np.mean((sgd1.predict(trainX) - trainY[:,0])**2, axis=0))
    print('Testing MSE:', np.mean((sgd1.predict(testX) - testY[:,0])**2, axis=0))
    print('SGD2 - stars')
    print('Training score:', sgd2.score(trainX, trainY[:,1]))
    print('Testing score:', sgd2.score(testX, testY[:,1]))
    print('Training MSE:', np.mean((sgd2.predict(trainX) - trainY[:,1])**2, axis=0))
    print('Testing MSE:', np.mean((sgd2.predict(testX) - testY[:,1])**2, axis=0))

    pickle.dump(sgd1, open("SGD_Comments.pkl", 'wb'))
    pickle.dump(sgd2, open("SGD_Stars.pkl", 'wb'))
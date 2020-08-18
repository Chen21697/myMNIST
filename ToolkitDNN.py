#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:32:32 2020

@author: yuwenchen
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist

#%%
def load_data():  
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    # x_test=np.random.normal(x_test)
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    # load training data and testing data
    (x_train, y_train), (x_test, y_test) = load_data()

    # define network structure
    model = Sequential()
    
    model.add(Dense(input_dim=28*28, units=500, activation='relu')) # input layer and layer1
    
    model.add(Dense(units=500, activation='sigmoid')) # layer1
    # model.add(Dense(units=500, activation='sigmoid')) # layer2
    model.add(Dense(units=10, activation='softmax')) # outputplayer

    # set configurations
    model.compile(loss='categorical_crossentropy',
                  optimizer="SGD", metrics=['accuracy'])

    # train model
    model.fit(x_train, y_train, batch_size=100, epochs=10)

    # evaluate the model and output the accuracy
    result = model.evaluate(x_test, y_test)
    print('Test Acc:', result[1])
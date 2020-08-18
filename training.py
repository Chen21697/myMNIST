#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:44:05 2020

@author: yuwenchen
"""
import numpy as np
import pandas as pd
from structure import *
from keras.datasets import mnist
from keras.utils import np_utils

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

#%%
epoch = 10
batch = 100
loss_history = []
lr = 0.001

(x_train, y_train), (x_test, y_test) = load_data()
datasetLength = len(x_train)
updatetime = datasetLength//batch # 10000/100 = 100

# initialize the parameters
x1 = x_train[:batch]
y1 = y_train[:batch]

#%%
if __name__ == '__main__':
    myDNN = DNN(None, None) # x y are from the first batch(done)
    
    for i in range(epoch): # calclate the loss in the end of each epoch(done)
        print("epoch", i+1)
        b_start, b_end = 0, batch
        
        for j in range(updatetime): # parameters will be updated each batch
            # import the batch training set first(done)
            x = x_train[b_start:b_end]
            y = y_train[b_start:b_end]
            myDNN.inputD = x
            myDNN.target = y
            myDNN.calculate()
            
            print(myDNN.lossFunc()/batch) # loss of this batch
            
            grads = myDNN.find_gradient() # find gradints for this batch
            
            # average gradints here
            #avg_grads = [temp / batch for temp in grads]
            
            myDNN.updatdParameter(grads, lr) # not yet
            
            b_start += batch
            b_end += batch
        

#%%
    #testing
    TP = 0   
    myDNN.inputD = x_test
    myDNN.target = y_test
    myDNN.calculate()
    
    actaul_index = np.argmax(y_test, axis=1)
    predicted_index = np.argmax(myDNN.layer3.a, axis=1)
    
    for p_index, a_index in zip (predicted_index, actaul_index):
        if p_index == a_index:
            TP += 1
        
    print("Fianl accuracy is:", TP/10000)
    
    

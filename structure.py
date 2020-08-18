#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:17:17 2020

@author: yuwenchen
"""
import numpy as np
np.random.seed(1)
#%%
class layer():
    def __init__(self, inputlen, neuron, activation, isOutput=None, isInput=None):
        
        self.isOutput = False
        self.isInput = False
        if isOutput:
            self.isOutput = True
        if isInput:
            self.isInput = True
        
        self.activation = activation
        self.neuron = neuron
        

        self.w = np.random.randn(inputlen, neuron) * 0.01
        self.b = np.full((1, neuron), 0)
               
#        self.w_mt = np.zeros(self.weight.shape)
#        self.w_vt = np.zeros(self.weight.shape)
#        self.b_mt = np.zeros(self.bias.shape)
#        self.b_vt = np.zeros(self.bias.shape)
#        self.sigma_w = np.zeros(self.weight.shape)
#        self.sigma_b = np.zeros(self.bias.shape)
#%%
class DNN():
    def __init__ (self, inputD, target):
        self.inputD = inputD
        self.target = target 
        
        self.layer1 = layer(784, 500, "relu", isInput=True)
        self.layer2 = layer(500, 500, "sigmoid")
        self.layer3 = layer(500, 10, "softmax", isOutput=True)
     
    def calculate(self):
        self.layer1.z = self.inputD @ self.layer1.w + self.layer1.b
        if self.layer1.activation is "relu":
            self.layer1.a = relu(self.layer1.z)
        elif self.layer1.activation is "sigmoid":
            self.layer1.a = sigmoid(self.layer1.z)
        elif self.layer1.activation is "softmax":
            self.layer1.a = softmax(self.layer1.z)
            
        self.layer2.z = self.layer1.a @ self.layer2.w + self.layer2.b
        if self.layer2.activation is "relu":
            self.layer2.a = relu(self.layer2.z)
        elif self.layer2.activation is "sigmoid":
            self.layer2.a = sigmoid(self.layer2.z)
        elif self.layer2.activation is "softmax":
            self.layer2.a = softmax(self.layer2.z)
            
        self.layer3.z = self.layer2.a @ self.layer3.w + self.layer3.b
        if self.layer3.activation is "relu":
            self.layer3.a = relu(self.layer3.z)
        elif self.layer3.activation is "sigmoid":
            self.layer3.a = sigmoid(self.layer3.z)
        elif self.layer3.activation is "softmax":
            self.layer3.a = softmax(self.layer3.z)

    def lossFunc(self):
        loss = np.sum(crossEntropy(self.target, self.layer3.a))
        return loss    
    
    
    def find_gradient(self):
        l3_backwardPass = softmax_crossEntropy_derivatives(self.target, self.layer3.a)
        
        l3_forwardPass = self.layer2.a.T
        l3_w_grad = l3_forwardPass @ l3_backwardPass
        l3_b_grad = np.sum(l3_backwardPass, axis=0, keepdims=True) # add gradient of all data
        
        l2_backwardPass = l3_backwardPass @ self.layer3.w.T * sigmoid_derivatives(self.layer2.z) # revised here
        l2_forwardPass = self.layer1.a.T
        l2_w_grad = l2_forwardPass @ l2_backwardPass
        l2_b_grad = np.sum(l2_backwardPass, axis=0, keepdims=True)
        
        l1_backwardPass = l2_backwardPass @ self.layer2.w.T * relu_derivatives(self.layer1.z) # revised here
        l1_forwardPass = self.inputD.T
        l1_w_grad = l1_forwardPass @ l1_backwardPass
        l1_b_grad = np.sum(l1_backwardPass, axis=0, keepdims=True)
        return [l1_w_grad, l2_w_grad, l3_w_grad, l1_b_grad, l2_b_grad, l3_b_grad]
    
    
    def updatdParameter(self, grads, lr):
        originalP = [self.layer1.w, self.layer2.w, self.layer3.w, self.layer1.b, self.layer2.b, self.layer3.b]        
        new_parameter = []
        
        for (param, gradient) in zip(originalP, grads):    
            # have to implemt optimizer here
            param = param - lr * gradient
            new_parameter.append(param)
            
        self.layer1.w = new_parameter[0]
        self.layer2.w = new_parameter[1]
        self.layer3.w = new_parameter[2]
        self.layer1.b = new_parameter[3]
        self.layer2.b = new_parameter[4]
        self.layer3.b = new_parameter[5]
        
         
#%%
def sigmoid(x):
    # x = x + 1e-5
    return 1 / (1 + np.exp(-x))

def softmax(x): # revised 
    shiftx = x - np.max(x, axis=1, keepdims=True)
    numerator = np.exp(shiftx)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    return numerator/denominator

def crossEntropy(y, s): # revised
    """Return the cross-entropy of vectors y and s.

    :type y: ndarray
    :param y: one-hot vector encoding correct class

    :type s: ndarray
    :param s: softmax vector

    :returns: scalar cost
    """
    return -np.log(s[np.where(y)])

def crossEntropy_derivatives(y, s): #(revised)
    """Return the gradient of cross-entropy of vectors y and s.

    :type y: ndarray
    :param y: one-hot vector encoding correct class

    :type s: ndarray
    :param s: softmax vector

    :returns: ndarray of size len(s)
    """
    return -y / s

def softmax_crossEntropy_derivatives(y, s):
    return s-y
    
def sigmoid_derivatives(x):
    return sigmoid(x)*(1-sigmoid(x))

def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)
    
def relu(Z):
    A = np.maximum(0, Z)
    return A

def relu_derivatives(x):
    return 1 * (x>0)

#%%
def adam(g, lr, iteration, mt, vt):

    b1 = 0.9
    b2 = 0.999
    e = 0.00000001
    
    mt = b1*mt + (1-b1)*g
    vt = b2*mt + (1-b2)*(np.power(g, 2))
    
    
    m_hat = mt / (1 - np.power(b1, iteration + 1))
    v_hat = vt / (1 - np.power(b2, iteration + 1))
    
    new_g = g - lr * m_hat / (np.sqrt(v_hat) + e)
    
    return new_g, mt, vt
#%%
def rmsProp(g, lr, iteration, sigma):
    
    a = 0.9
    epsilon = 1e-8
    
    if iteration == 1:
        sigma = g
    else:
        sigma = np.sqrt(a*np.power(sigma, 2) + (1-a)*np.power(g, 2))
    
    
    new_g = g - lr*g/(sigma + epsilon)
    
    return new_g, sigma


import numpy as np
import random

def signold(x):
    return 1/(1+x)
def signold_der(x):
    return x*(1-x)

class NN:
    def __init__(self,inputs):
        random.seed(1)
        self.inputs = inputs
        self.l = len(self.inputs)
        self.li = len(self.inputs[0])
        
        self.wl = np.random.random((self.li,self.l))
        self.wh = np.random.random((self.l,1))
    
    def think(self,inp):
        s1 = signold(np.dot(inp,self.wl))
        s2 = signold(np.dot(s1,self.wh))
        return s2
    
    def train(self,inputs,outputs,it):
        for i in range(it):
            l0 = inputs
            l1 = signold(np.dot(l0,self.wl))
            l2 = signold(np.dot(l1,self.wh))
            
            l2_err = outputs - l2
            l2_delta = np.multiply(l2_err, signold_der(l2))
            
            l1_err = np.dot(l2_delta,self.wh.T)
            l1_delta = np.multiply(l1_err, signold_der(l1))
            
            self.wh += np.dot(l1.T,l2_delta)
            self.wl += np.dot(l0.T,l1_delta)
            

inputs = np.array([[1,0,1],[1,1,0],[0,0,1]])
outputs = np.array([[1],[1],[0]])
outputs1 = np.array([[0],[1],[0]])
outputs2 = np.array([[1],[0],[1]])
n = NN(inputs)
print("Part-1>> Before Training")
print(n.think(inputs))
n.train(inputs, outputs, 100000)
print("After Training")
print(n.think(inputs))
print("Part-2>> Before Training")
print(n.think(inputs))
n.train(inputs, outputs1, 100000)
print("After Training")
print(n.think(inputs))
print("Part-3>> Before Training")
print(n.think(inputs))
n.train(inputs, outputs2, 100000)
print("After Training")
print(n.think(inputs))

            
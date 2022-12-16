#!/usr/bin/env python
# coding: utf-8


class NeuralNetwork():
    
    """
    Parameters: 
    
    -------------------------------------------
    Step1: Initialization
    
    Example: 
    test_normal = NeuralNetwork(layersize = [10,10],init_value = 0.01,learningrate = 0.1, optimizer = "normal", activation = "sigmoid")
    -------- 
    
    layersize: hidden layer neural size, e.g. [256,256] means two hidden layer, 256 for each;
    optimizer: adam or normal;
    activation: "linear" or "sigmoid";
    learning rate: float; 
    loss bound: float, which control the loss to below this threhold;
    stop rate: float, which control the sample stop when it converge.
    
    --------------------------------------------
    Step2: Fit
    
    Example: 
    test_normal.fit(x_t,y_t)
    --------
    
    Input: X: array[[],[],...[]], 
           Y: array[[],[],...[]]. 
    
    --------------------------------------------
    Step3: Train
    
    Example: 
    1) for very simple case: 
    test_normal.simple_train(echos = 20000, verbose = 5000)
    2) for picture case which need to draw batchsize: 
    test_normal.train(batchsize = 3, more_echos = 40000-image1_nn1.echos, verbose = 100)
    ---------
    
    more_echos: integer, which ask the model to train more echos based on previous training;
    verbose: integer, defualt, 100, which output loss every 100 echos;
    batchsize: integer, default 3, define how many samples training once in a echo;
    inner_iteration: integer, default, float("inf"), echo stop when while loops reach to inner_iteration.
    
    --------------------------------------------
    Step4: Prediction
    
    Example: 
    1) for simple test case: 
    pred = test_normal.predict(x_t, show_pred = True)
    
    2) for image: 
    image1_nn2.predict(x1_s,ishape = (100,76),setting = True, ishow = True)
    ----------
    x_s: predictors, array[[],[],...[]];
    ishape: image shape, e.g. [100,76], input if it is image data; 
    show_pred: boolean, default, False, if show the prediction value;
    ishow: boolean, default, False, if show the picture; 
    rescale_value: integer, for image plotting setting, default, 255; 
    setting: boolean, default False, if output the model parameter setting; 
    image_type: string, default "gray", control the type of output image; 
    plt_title: string, if "echos", then output the echos as title; if other string, then title as your defined string.
    
    
    Step5: Prediction Loss: 
    
    Example: 
    nn.pred_loss(self, y_s, loss_type = "both")
    ----------
    
    y_s: target, array[[],[],...[]]; 
    loss_type: string, "average", "total", "both"
    
    """
    
    def __init__(self,layersize, optimizer,init_value = 0.01, activation = None, learningrate = 0.0001, lossbound = 0.01, stoprate = 1e-5):
        # initializing data
        self.layer = len(layersize)+1 # an list that consist of layersize for every layer
        self.layersize = layersize # hidden layersize, not include input of layer1 and output of layern
        self.learningrate = learningrate
        # fitting data
        self.params = {}
        self.store = {} # store the output value for every layer for every x
        self.inputdim = None
        self.inputsize = None
        self.inputX = None
        self.target = None
        # training data
        self.dev = {}
        self.method = optimizer
        self.activation = activation
        self.init_value = init_value
        self.lossbound = lossbound
        self.stoprate = stoprate
        self.pred = None
        self.echos = 0
        self.predicted = False
        self.batchsize = None
    
    def forward(self,X): # check 
        # here x is element 
        # use function linear forward, sigmoid forward
        self.store["res"+str(0)] = np.array(X)
        for layer in range(1,self.layer+1):
            self.store["f"+str(layer)] = self.linear_forward(self.store["res"+str(layer-1)],layer)
            self.store["res"+str(layer)] = self.store["f"+str(layer)]
            if self.activation == "sigmoid":
                self.store["sig"+str(layer)] = self.sigmoid_forward(self.store["f"+str(layer)])
                self.store["res"+str(layer)] = self.store["sig"+str(layer)]
        
    def backward(self,x,y):
        # calculate gradient 
        self.dev["b"+str(self.layer)] = np.multiply(self.loss_backward(x,y),self.sigmoid_backward(self.layer))
        self.dev["W"+str(self.layer)] = np.dot(np.transpose(self.store["res"+str(self.layer-1)]),self.dev["b"+str(self.layer)])
        
        for l in range(self.layer-1,0,-1):
            self.dev["b"+str(l)] = np.transpose(np.dot(self.params["W"+str(l+1)],np.transpose(self.dev["b"+str(l+1)])))*self.sigmoid_backward(l)
            self.dev["W"+str(l)] = np.dot(np.transpose(self.store["res"+str(l-1)]),self.dev["b"+str(l)])
        
        for l in range(1,self.layer+1):
            self.dev["b"+str(l)] = sum(self.dev["b"+str(l)])/len(x)
            self.dev["W"+str(l)] = self.dev["W"+str(l)]/len(x)
            
    def sigmoid_forward(self,X): #check
        sig = 1/(1+np.exp(-X))
        return sig
    
    def sigmoid_backward(self,layer): #check
        if self.activation == "sigmoid":
            sig = self.store["sig"+str(layer)]
            sig_d = np.multiply(sig,(1-sig))
        else:
            sig_d = 1
        return sig_d
        
    def linear_forward(self,X,layer): # check
        W = self.params["W"+str(layer)]
        b = self.params["b"+str(layer)]
        f = np.dot(X,W) + np.tile(b, (len(X),1))
        return f
    
    def loss(self,x,y):
        pred = self.store["res"+str(self.layer)]
        loss = 1/2*(pred - y)**2
        return loss
    
    def loss_backward(self,x,y): #?
        pred = self.store["res"+str(self.layer)]
        los_d = pred-y
        return los_d
        
    def update_params(self):
        if self.method == "normal":
            for l in range(1,self.layer+1):
                self.params["W"+str(l)] -= self.learningrate*self.dev["W"+str(l)]
                self.params["b"+str(l)] -= self.learningrate*self.dev["b"+str(l)]
        
        if self.method == "adam": 
            for l in range(1,self.layer+1):

                self.m["W"+str(l)]= self.beta1*self.m["W"+str(l)]+(1-self.beta1)*self.dev["W"+str(l)]
                self.v["W"+str(l)]= self.beta2*self.v["W"+str(l)]+(1-self.beta2)*(self.dev["W"+str(l)])**2
                self.params["W"+str(l)] -= self.learningrate*self.m["W"+str(l)]/(np.sqrt(self.v["W"+str(l)])+1e-4)
                
                self.m["b"+str(l)]= self.beta1*self.m["b"+str(l)]+(1-self.beta1)*self.dev["b"+str(l)]
                self.v["b"+str(l)]= self.beta2*self.v["b"+str(l)]+(1-self.beta2)*(self.dev["b"+str(l)])**2
                self.params["b"+str(l)] -= self.learningrate*self.m["b"+str(l)]/(np.sqrt(self.v["b"+str(l)])+1e-4)
        
    def fit(self,X,Y): # get in data 
        self.inputdim = len(X[0])
        self.inputsize = len(X)
        self.inputX = X
        self.target = Y
        self.outputdim = len(Y[0])
        self.layersize.append(len(Y[0]))
        
        # initial params for adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.m = {}
        self.v = {}
        
        # initialize the parameters:
        self.params["W1"] = np.random.rand(self.inputdim,self.layersize[0])*self.init_value
        self.params["b1"] = np.random.rand(1,self.layersize[0])*self.init_value
        if self.method == "adam":
            self.m["W"+str(1)] = np.zeros((self.inputdim,self.layersize[0]))
            self.v["W"+str(1)] = np.zeros((self.inputdim,self.layersize[0]))
            self.m["b"+str(1)] = np.zeros((1,self.layersize[0]))
            self.v["b"+str(1)] = np.zeros((1,self.layersize[0]))
        
        for k in range(2,self.layer+1):
            self.params["W"+str(k)] = np.random.rand(self.layersize[k-2],self.layersize[k-1])*self.init_value
            self.params["b"+str(k)] = np.random.rand(1,self.layersize[k-1])*self.init_value
            if self.method == "adam":
                self.m["W"+str(k)] = np.zeros((self.layersize[k-2],self.layersize[k-1]))
                self.v["W"+str(k)] = np.zeros((self.layersize[k-2],self.layersize[k-1]))
                self.m["b"+str(k)] = np.zeros((1,self.layersize[k-1]))
                self.v["b"+str(k)] = np.zeros((1,self.layersize[k-1]))
                
                
        self.full = []
        for i in range(self.inputsize):
            a = list(self.inputX[i])
            a.extend(list(self.target[i]))
            self.full.append(a)
            
        import pandas as pd
        self.full = pd.DataFrame(self.full)
        
        return self
        
    def train(self, more_echos, verbose = 100, batchsize = 3, inner_iteration = float("inf")):
        echo = self.echos
        self.batchsize = batchsize
        for e in range(more_echos):
            self.echos += 1
            shuffled = self.full.sample(frac=1)
            xi = np.array(shuffled.iloc[0:self.batchsize,0:self.inputdim])
            yi = np.array(shuffled.iloc[0:self.batchsize,self.inputdim:self.inputdim+self.outputdim])
            self.forward(xi)
            loss = self.loss(xi,yi)
            l_p = 0
            l_c = sum(sum(loss))/self.batchsize
            times = 0
            while (l_c - l_p > self.stoprate or l_c > self.lossbound) and times < inner_iteration:
                times += 1
                l_p = l_c
                self.backward(xi,yi)
                self.update_params()
                self.forward(xi)
                loss = self.loss(xi,yi)
                l_c = sum(sum(loss))/self.batchsize
                
            if (e+1)%verbose == 0:
                print(f"echo:{self.echos}/{echo+more_echos},loss:{sum(sum(loss))/self.batchsize}")
        print(f"Note: Final output would be based on {self.echos} iterations trained")
                
    def simple_train(self, echos, verbose = 100):
        self.predicted = True
        self.echos = echos
        for e in range(echos):
            self.forward(self.inputX)
            loss = self.loss(self.inputX,self.target)
            self.backward(self.inputX,self.target)
            self.update_params()
            if (e+1)%verbose == 0:
                print(f"echo:{e+1}/{self.echos},total loss:{sum(sum(loss))/self.inputsize}")
                
    def predict(self, x_s, ishape = None, show_pred = False,ishow = False, rescale_value = 255, setting = False, image_type = "gray", plt_title = "echos"):
        self.predicted = True
        self.forward(x_s)
        self.pred = self.store["res"+str(self.layer)]
        if ishow:
            import matplotlib.pyplot as plt
            a = self.pred*rescale_value
            a = np.reshape(a,ishape)
            if image_type == "gray":
                plt.imshow(a, cmap = "gray")
            elif image_type == "uint8":
                plt.imshow(a.astype(np.uint8))
            
            if plt_title == "echos":
                plt.title(f"echo{self.echos}")
            else:
                plt.title(plt_title)
                
        if setting:
            self.show_setting()
            print("---------------------------------------------")
        print(f"Note: The prediction based on {self.echos} echos training.")
        if show_pred:
            return self.pred
            
        
    def pred_loss(self, y_s, loss_type = "both"):
        if self.predicted:
            tloss = sum(sum(self.loss(None,y_s)))
            aloss = tloss/len(y_s)
            if loss_type == "both":
                return [tloss, aloss]
            elif loss_type == "total":
                return tloss
            elif loss_type == "average":
                return aloss
        else:
            print("Please predict first.")
            
    def show_setting(self):
        print("Hyperparams Setting:")
        print("---------------------------------------------")
        print(f"Hidden layer: {self.layersize}")
        print(f"Batchsize: {self.batchsize}")
        print(f"Learning rate:{self.learningrate}")
        print(f"Optimizer:{self.method}")
        print(f"Activation: {self.activation}")
        print(f"Loss bound:{self.lossbound}")
        print(f"Stop rate:{self.stoprate}")



# In[ ]:


# Before Inputing data: 
## we need to normalize the X and Y: 

import scipy.io
mat = scipy.io.loadmat('nn_data.mat')

# set random seed: 
import numpy as np
np.random.seed(67)

# Normalize image1 data:
x11 =  (mat["X1"].T[0] - np.mean(mat["X1"].T[0]))/99
x12 = (mat["X1"].T[1] - np.mean(mat["X1"].T[1]))/75
import pandas as pd
df1 = pd.DataFrame(x11,x12).reset_index()
x1_s = np.array(df1.iloc[:,0:2])
y1_s = mat["Y1"]/255

# Normalize image1 data:
x21 =  (mat["X2"].T[0] - np.mean(mat["X2"].T[0]))/132
x22 = (mat["X2"].T[1] - np.mean(mat["X2"].T[1]))/139
import pandas as pd
df2 = pd.DataFrame(x21,x22).reset_index()
x2_s = np.array(df2.iloc[:,0:2])
y2_s = mat["Y2"]/255



# ------------------------Image 1-----------------------------


## fitting data
np.random.seed(67)
image1_nn1 = NeuralNetwork(layersize = [256,256],init_value = 0.01,learningrate = 0.001, optimizer = "adam", activation = "sigmoid",lossbound = 0.01, stoprate = 2e-5)
image1_nn1.fit(x1_s,y1_s)


## training data
image1_nn1.train(batchsize = 3, more_echos = 40000-image1_nn1.echos, verbose = 100)


## showing prediction
image1_nn1.predict(x1_s,ishape = (100,76),setting = True, ishow = True,image_type = "gray")


## prediction loss
image1_nn1.pred_loss(y1_s,loss_type = "both")


# ------------------------Image 2--------------------------------

## fitting data
np.random.seed(67)
image2_nn1 = NeuralNetwork(layersize = [256,256],init_value = 0.0001,learningrate = 0.001, optimizer = "adam", activation = "sigmoid",lossbound = 0.05, stoprate = 5e-5)
image2_nn1.fit(x2_s,y2_s)


## training data
image2_nn1.train(batchsize = 1, more_echos = 40000-image2_nn1.echos, verbose = 100, inner_iteration = 5000)


## showing prediction
image2_nn1.predict(x2_s,ishape = (133,140,3),setting = True, ishow = True,image_type = "uint8")


## prediction loss
image2_nn1.pred_loss(y1_s,loss_type = "both")


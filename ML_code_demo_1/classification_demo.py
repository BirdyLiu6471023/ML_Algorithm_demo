#!/usr/bin/env python
# coding: utf-8

# # Homework 1 Question 4

# ## Stemming and Bag of words

# In[ ]:


import nltk
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


# In[4]:


stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))


# In[5]:


# creating the dicionary
import os
import string

dic = set()

for file in os.listdir("enron1/ham"):
    if file.endswith(".txt"):
        f = open("enron1/ham/"+file,"r",encoding = "ISO-8859-1")
        for line in f:
            for word in line.split(" "):
                word = word.lower().rstrip("\n")
                word = stemmer.stem(word)
                if word not in string.punctuation: # do we need to care about ","...
                    if word not in stop_words:
                        if word not in dic:
                            dic.add(word)
                    
for file in os.listdir("enron1/spam"):
    if file.endswith(".txt"):
        f = open("enron1/spam/"+file,"r",encoding = "ISO-8859-1")
        for line in f:
            for word in line.split(" "):
                word = word.lower().rstrip("\n")
                word = stemmer.stem(word)
                if word not in string.punctuation: # do we need to care about ","...
                    if word not in stop_words:
                        if word not in dic:
                            dic.add(word)


# In[6]:


# transforming the set into list
dicl = list(dic)
len(dicl)


# In[7]:


# creating the dataset:

dataset = []

for file in os.listdir("enron1/ham"):
    if file.endswith(".txt"):
        f = open("enron1/ham/"+file,"r",encoding = "ISO-8859-1")
        record = [0]*len(dicl)
        record.append(0) # label for ham = 0 ; label for spam = 1 
        for line in f:
            for word in line.split(" "):
                word = word.lower().rstrip("\n")
                word = stemmer.stem(word)
                if (word not in string.punctuation) and (word not in stop_words):
                    record[dicl.index(word)] +=1
        dataset.append(record)


# In[15]:


for file in os.listdir("enron1/spam"):
    if file.endswith(".txt"):
        f = open("enron1/spam/"+file,"r",encoding = "ISO-8859-1")
        record = [0]*len(dicl)
        record.append(1) # label for ham = 0 ; label for spam = 1 
        for line in f:
            for word in line.split(" "):
                word = word.lower().rstrip("\n")
                word = stemmer.stem(word)
                if (word not in string.punctuation) and (word not in stop_words):
                    record[dicl.index(word)] +=1
        dataset.append(record)


# In[16]:


len(dataset)


# In[17]:


# writing it into a .csv file, so I do not need to stemming and so on again and again. 
with open('full.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(dataset)


# ## Preprocess the dataset

# In[3]:


import pandas as pd
df = pd.read_csv("full.csv",header = None)


# In[4]:


def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices


# In[5]:


# Before I go into this Method, I would cut the words that only used once in the dataset, 
# which could not proviod much information.
al = list(df.sum(axis = 0) == 1)
len(al)


# In[6]:


drop_list = find_indices(al, 1)
clean = df.drop(drop_list,axis = 1)
len(clean.columns)


# In[427]:


import random
random.seed(42)
shuffled = clean.sample(frac=1)
train = shuffled.iloc[0:3620]
test = shuffled.iloc[3620:5172]
#42791 is the name of label for this dataset:
y_train = train[42791]
X_train = train.drop(42791,axis = 1)
y_test = test[42791]
X_test = test.drop(42791,axis = 1)


# ## Naive Bayes

# In[227]:


import numpy as np
import pandas as pd
class NaiveBayes:
    def __init__(self, k=0.5):
        self.k = k # laplace smoothing
        self.words_prob0 = []
        self.words_prob1 = []
        self.prior_cat0 = 0
        self.prior_cat1 = 0
        self.y_pred = []
        
    def prior(self,y): # we put the label in the last column 
        prior_cat1 = y.mean()
        prior_cat0 = 1-prior_cat1
        return prior_cat0, prior_cat1
    
    def words_num(self,X,y): # count the number of occurence of word groupby label
        X = X.copy()
        X["label"] = y
        grouped = X.groupby("label").sum()
        words_num0 = sum(grouped[grouped.index == 0].values)
        words_num1 = sum(grouped[grouped.index == 1].values) # sum() is to reduce the dimension, which with [[0,0,..]]
        return words_num0, words_num1 # the output is two arrays
    
    def words_prob(self,X,y):
        words_num0, words_num1 = self.words_num(X,y)
        words_prob0 = (words_num0+self.k)/sum(words_num0+self.k)
        words_prob1 = (words_num1+self.k)/sum(words_num1+self.k)
        return words_prob0,words_prob1 # the output is two arrays
        
    def fit(self,X,y):
        self.words_prob0, self.words_prob1 = self.words_prob(X,y)
        self.prior_cat0, self.prior_cat1 = self.prior(y)
        
    def pred(self,X_test):
        p_log0, p_log1 = np.log(self.words_prob0), np.log(self.words_prob1)
        f0 = np.matmul(X_test,p_log0)+np.log(self.prior_cat0)
        f1 = np.matmul(X_test,p_log1)+np.log(self.prior_cat1)
        y_pred = f0<f1
        self.y_pred = y_pred.astype(int)
    
    def acc(self,y_test):
        acc_list = y_test == self.y_pred
        acc = acc_list.mean()
        return acc
        
        
        


# In[228]:


nb = NaiveBayes()


# In[229]:


nb.fit(X_train,y_train)


# In[230]:


nb.pred(X_train)
nb.acc(y_train)


# In[231]:


nb.pred(X_test)
nb.acc(y_test)


# In[460]:


def compareNB(size):
    random.seed(1023)
    shuffled = clean.sample(frac=1)
    train_5h = shuffled.iloc[0:size]
    X_train_5h = train_5h.drop(42791,axis = 1)
    y_train_5h = train_5h[42791]
    test_p = shuffled.iloc[-1000:len(shuffled)]
    y_test_p = test_p[42791]
    X_test_p = test_p.drop(42791,axis = 1)
    nb_5h = NaiveBayes()
    nb_5h.fit(X_train_5h,y_train_5h)
    nb.pred(X_test_p)
    
    return nb.acc(y_test_p)


# In[464]:


a = []
s = [200,500,1000,1500,2000,3000]
for i in s:
    a.append(compareNB(i))


# In[480]:


import matplotlib.pyplot as plt
f,ax = plt.subplots()
plt.plot(s,a)
plt.ylim((0.9,1))
plt.xlabel("training size")
plt.ylabel("accuracy")


# ## Nearest Neighbour

# In[554]:


class KDTreeNode:
    def __init__(self, point=[], spt_idx = None, left=None, right=None):
        self.point = point
        self.spt_idx = spt_idx
        self.left = left
        self.right = right


# In[643]:


class NearestNeighbour:
    """
    Reference: lec2 Reading: An introductory tutotial on kd-tree. 
    """
    def __init__(self,p=2,depth=5):
        # p is the norm, determin how to calculate the distance:
        self.p = p
        self.depth = depth
        self.kd_tree = None
        self.hr_range = np.array([])
        self.split_ord = []
        self.max_dist_sqd = float("inf")
        self.nearest = []
        self.prediction = []
        
    def rank_simple(self,v): 
        """
        This is to rank dimension to split, used in variance matrix.
        Input: vector
        Output: a tuple, (m,i), m = variance, i = the corresponding index
        Usage: self.hyperrectangle
        """
        des = []
        for i,m in enumerate(v):
            des.append((m,i))
            
        des.sort()
        des.reverse()
        return des
        
    def hyperRectangle(self,X): 
        """
        hr_range: variance for every words
        split_ord: it would give the order of spliting the dimension
        Usage: self.fit()
        """
        self.hr_range = np.array(X.var(axis = 0))
        self.split_ord = self.rank_simple(self.hr_range)
        
    def middlePoint_split(self, exset, spt_idx):
        """
        This function is to find the middle points along a direction
        1) sorting the points according to its spt_idx 
        2) find the n/2th point
        """
        n = len(exset)
        mid = n//2 
        
        col = exset.columns[spt_idx]
        df.sort_values(by = col)
        mid_point = exset.iloc[mid,:]
        exset_left = exset.iloc[0:mid,:]
        exset_right = exset.iloc[mid+1:n,:]
        
        return mid_point,exset_left,exset_right
        
    def kd_tree_built(self,exset,split,dep):
        if len(exset)==0:
            return
        if dep > self.depth:
            return
    
        """ 
        The root of kd-tree is found by:
        Firstly, find the dimension with longest variance range;
        Secondly, the middle points along this dimension. 
        -----------------------------------------------------
        Using KDTreeNode built before to construct the kd-tree. 
        -----------------------------------------------------
        Thinking: for time efficiency, we might could control the depth of the tree, for example depth = 8 or 10. 
        """
        spt_idx = self.split_ord[split][1]
        ex, exset_left,exset_right = self.middlePoint_split(exset,spt_idx)
        root = KDTreeNode(point = ex,spt_idx = spt_idx)
        root.left = self.kd_tree_built(exset_left,split+1,dep+1)
        root.right = self.kd_tree_built(exset_right,split +1,dep+1)
        return root
        
        
    def fit(self,X,y):
        """
        This is the first step to do when we deal with the X and y.
        """
        self.hyperRectangle(X)
        exset = X.copy()
        exset["label"] = y
        self.kd_tree = self.kd_tree_built(exset,0,0)
        
    def NNSearch(self,target, kd_tree): 
        """
        NNSearch would find the nearest point of the target. 
        """
        if not kd_tree:
            return 
        
        s = self.kd_tree.spt_idx
        pivot = kd_tree.point
        
        ## calculate the distance between pivot and target: 
        comp_dist = np.subtract(list(target),list(pivot)[0:len(pivot)-1])
        if self.p == 1:
            dist_sqd = sum(abs(comp_dist))
        elif self.p == 2:
            dist_sqd = sum(comp_dist**2)
        elif self.p == "inf":
            dist_sqd = max(abs(comp_dist))
            
        ## store the distance in self.max_dist_sqd and points found in self.nearest
        if  dist_sqd <=self.max_dist_sqd:
            self.max_dist_sqd = dist_sqd
            self.nearest.append(pivot) #update the nearest point is the pivot of the tree.
        
        if target.iloc[s] <= pivot.iloc[s]:
            nearer_kd = kd_tree.left
            further_kd = kd_tree.right
        else:
            nearer_kd = kd_tree.right
            further_kd = kd_tree.left
            
        self.NNSearch(target,nearer_kd)
        if self.p != "inf":
            if abs(target.iloc[s] - pivot.iloc[s])**(self.p) < dist_sqd:
                self.NNSearch(target,further_kd)
                
        else: 
            if abs(target.iloc[s] - pivot.iloc[s]) < dist_sqd:
                self.NNSearch(target,further_kd)
                
    def pred(self,X_test,num_of_nn=1):
        """
        parameters to use: 
        num_of_nn (how many nearest neighbour take into account)
        self.NNSearch for the nearest points around target
        output: 
        the label of k nearest neighbors's mean value, label in nearest["label"]
        """
        l = []
        self.prediction = []
        for i in range(0,len(X_test)):
            self.NNSearch(X_test.iloc[i,:],self.kd_tree)
            
            if len(self.nearest) >= num_of_nn:
                k = num_of_nn 
            else:
                k = len(self.nearest)

            k_points = self.nearest[-k:]
            for x in range(0,k):
                l.append(k_points[x]["label"])
                
            if sum(l)<= k/2:
                self.prediction.append(0)
            else:
                self.prediction.append(1)
            self.nearest = []
            
    def acc(self,y_test):
        acc_list = y_test == np.array(self.prediction)
        acc = acc_list.mean()
        return acc

            


# #### Before fitting ...

# In[708]:


# we need to reduce demension to make the number of dataset dimension < training sample size set as 3000
a = sorted(list(np.array(clean.var(axis= 0))))
l = range(0,len(a))
plt.plot(l,a)
#plt.xlim((17500,21000))
plt.xlabel("index")
plt.ylabel("variance of each words")


# ## Depth = 5, training size = 3620 (selected # column = 2434 )

# In[865]:


# drop the columns according to the variance : 
l = clean.var(axis = 0)
drop_list = []
up = l[l>0.5].reset_index()
drop_list = list(up['index'])
lo = l[l<0.013435928223061857].reset_index()
drop_list.extend(list(lo['index']))
if 42791 in drop_list:
    drop_list.pop(drop_list.index(42791))
clean_knn = clean.drop(drop_list,axis = 1)

# randomize the sample:
random.seed(4728)
shuffled = clean_knn.sample(frac=1)
train = shuffled.iloc[0:3620]
test = shuffled.iloc[3620:5172]
#42791 is the name of label for this dataset:
y_train = train[42791]
X_train = train.drop(42791,axis = 1)

# draw 500 test cases from test set, which could be more time efficient
test = test.sample(frac = 0.33)
y_test = test[42791]
X_test = test.drop(42791,axis = 1)


# In[866]:


len(clean_knn.columns)


# In[867]:


knn_p1 = NearestNeighbour(p=1)
knn_p1.fit(X_train,y_train)


# In[868]:


knn_p1.pred(X_test,num_of_nn = 1)
knn_p1.acc(y_test)


# In[869]:


knn_p1.pred(X_test,num_of_nn = 3)
knn_p1.acc(y_test)


# #### Cutting Traing Size to See accuracy score...

# In[ ]:


#size = 543


# In[886]:


knn_p1_1 = NearestNeighbour(p=1)
knn_p1_1.fit(X_train.iloc[0:543,:],y_train.iloc[0:543])


# In[883]:


knn_p1_1.pred(X_test,num_of_nn = 1)
knn_p1_1.acc(y_test)


# In[888]:


knn_p1_1.pred(X_test,num_of_nn = 3)
knn_p1_1.acc(y_test)


# In[ ]:


# size = 1086


# In[872]:


knn_p1_2 = NearestNeighbour(p=1)
knn_p1_2.fit(X_train.iloc[0:1086,:],y_train.iloc[0:1086])


# In[873]:


knn_p1_2.pred(X_test,num_of_nn = 1)
knn_p1_2.acc(y_test)


# In[874]:


knn_p1_2.pred(X_test,num_of_nn = 3)
knn_p1_2.acc(y_test)


# In[ ]:


# size = 2353


# In[878]:


knn_p1_3 = NearestNeighbour(p=1)
knn_p1_3.fit(X_train.iloc[-2353:len(X_train),:],y_train.iloc[-2353:len(X_train)])


# In[879]:


knn_p1_3.pred(X_test,num_of_nn = 1)
knn_p1_3.acc(y_test)


# In[880]:


knn_p1_3.pred(X_test,num_of_nn = 3)
knn_p1_3.acc(y_test)


# In[892]:


k3 = [knn_p1_1.acc(y_test),knn_p1_2.acc(y_test), knn_p1_3.acc(y_test),knn_p1.acc(y_test)]
s = [543,1086,2353,3620]
plt.plot(s,k3)
plt.xlabel("training size")
plt.ylabel("accuracy score")
plt.title("KNN (k=3,depth=5)")


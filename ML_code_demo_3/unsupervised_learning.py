#!/usr/bin/env python
# coding: utf-8

# In[ ]:

## Question 2 and 3 code submission


# Question 2 code:
# k_means method is implemented by myself:

class K_means_V1():
    """
    How to run code: 
    > sample data: 
    X, y = make_circles(n_samples=1000, factor=0.3, noise=0.05, random_state=0)
    
    > training:
    1. km_t1 = K_means_V1(k = 2)
    -- k: integer, num of cluster you want to train. 
    
    2. km_t1.fit(X)
    
    3. km_t1.train(echos = 10, transform = True, r =6) 
    -- echos: integer, num of iterations;
    -- transform: boolean, True if you need to transform it into felxible version; 
    -- r: integer, num of neighbors you create the lapaliance matrix.
    
    """
    
    def __init__(self, k):
        self.k = k
        self.total_distance = None
        
    def center_assign(self):
        self.cluster = {}
        for i in range(self.k):
            self.cluster[i] = []
            
        self.total_distance = 0
        for x in self.inputX:
            d = float("inf")
            ci = None
            for i in range(self.k):
                cur_d = sum((x-self.centers[i])**2)
                cur_i = i
                if cur_d < d:
                    d = cur_d
                    ci = cur_i
                    
            self.cluster[ci].append(x)
            self.total_distance += d
            
    def update_center(self):
        for i in range(self.k):
            if self.cluster[i]:
                c = np.array(self.cluster[i])
                self.centers[i] = np.array([np.mean(x) for x in c.T])
        self.centers = np.array(self.centers)
    
    
    def nearest_neighbors(self, r):
        # Adj_mat = squareform(pdist(X, metric='euclidean'))
        Adj_mat = []
        for x in self.inputX:
            d = (self.inputX - x)**2
            distance = 0
            for i in range(self.dim):
                distance += d[:,i]
            Adj_mat.append(distance)
        Adj_mat = np.array(Adj_mat)
        
        
        W = np.zeros(Adj_mat.shape)
        
        Adj_sort = np.argsort(Adj_mat, axis=1)
        # Set the weight (i,j) to 1 when either i or j is within the k-nearest neighbors of each other
        for i in range(Adj_sort.shape[0]):
            W[i,Adj_sort[i,:][:(r+1)]] = 1
        return W
            
    
    def transform(self, r):
        self.W = np.array(self.nearest_neighbors(r))
        self.D = np.diag(np.sum(self.W, axis=1))
        self.L = self.D - self.W
        
        self.Evalue, self.Evector = np.linalg.eig(self.L)
        
        ind = np.argsort(np.linalg.norm(np.reshape(self.Evalue, (1, len(self.Evalue))), axis=0))
        self.V = np.real(self.Evector[:, ind[:self.k]])
        
        self.centers = []
        for i in range(self.dim):
            self.centers.append(self.V[np.random.randint(self.inputsize)])
        self.centers = np.array(self.centers).T
        
        return self
    
    def transform_center_assign(self):
        self.vcluster = {}
        self.cluster = {}
        for i in range(self.k):
            self.vcluster[i] = []
            self.cluster[i] = []
            
        for vi in range(self.inputsize):
            d = float("inf")
            ci = None
            for i in range(self.k):
                cur_d = sum((self.V[vi]-self.centers[i])**2)
                cur_i = i
                if cur_d < d:
                    d = cur_d
                    ci = cur_i
            
            self.cluster[ci].append(self.inputX[vi])
            self.vcluster[ci].append(self.V[vi])
        
    def transform_update_center(self): 
        for i in range(self.k):
            if self.vcluster[i]:
                c = np.array(self.vcluster[i])
                self.centers[i] = np.array([np.mean(x) for x in c.T])
        self.centers = np.array(self.centers)
        
    
    def fit(self, X):
        self.inputX = np.array(X)
        self.dim = len(self.inputX[0])
        self.inputsize = len(self.inputX)
        self.range = [max(self.inputX[:,c])-min(self.inputX[:,c]) for c in range(self.dim)]
        self.centers = []
        for i in range(self.dim):
            self.centers.append(np.random.rand(1,self.k)[0]*self.range[i])
        self.centers = np.array(self.centers).T
        
        self.cluster = {}
        for i in range(self.k):
            self.cluster[i] = []
            
        return self
        
    def train(self, echos, verbose = 0, transform = False, r = 2):
        if not transform: 
            self.center_assign()
            print(f"Total distance of initialization is: {self.total_distance}")
            for e in range(echos):
                self.update_center()
                self.center_assign()
                if verbose == 1:
                    print(f"Total distance is: {self.total_distance}")
            print(f"After {echos} train, total distance is: {self.total_distance}")
            
        else: 
            self.transform(r)
            self.transform_center_assign()
            for e in range(echos):
                self.transform_update_center()
                self.transform_center_assign()


# In[ ]:


## 1) non_transformation example: 
from sklearn.datasets import make_circles
np.random.seed(42)
X, y = make_circles(n_samples=1000, factor=0.3, noise=0.05, random_state=0)
km_t1 = K_means_V1(2)
km_t1.fit(X)
km_t1.train(echos = 10, transform = False)


# plot the k_mean sperate
X_1 = np.array(km_t1.cluster[0])
y_1 = np.ones((len(X),1))
X_2 = np.array(km_t1.cluster[1])
y_2 = np.zeros((len(X),1))

_, (ax) = plt.subplots(sharex=True, sharey=True, figsize=(6, 4))
ax.scatter(X_1[:, 0], X_1[:, 1], c="purple")
ax.scatter(X_2[:, 0], X_2[:, 1], c= "orange")
ax.set_ylabel("Feature #1")
ax.set_xlabel("Feature #0")
ax.set_title(f" K_means Seprate")


# In[ ]:


## 2) transfromation example: 
from sklearn.datasets import make_circles
np.random.seed(42)
X, y = make_circles(n_samples=1000, factor=0.3, noise=0.05, random_state=0)
km_t1 = K_means_V1(2)
km_t1.fit(X)
r = 6
km_t1.train(echos = 10, transform = True, r = r )


# plot the k_mean with transformation sperate
X_1 = np.array(km_t1.cluster[0])
y_1 = np.ones((len(X),1))
X_2 = np.array(km_t1.cluster[1])
y_2 = np.zeros((len(X),1))

_, (ax) = plt.subplots(sharex=True, sharey=True, figsize=(6, 4))
ax.scatter(X_1[:, 0], X_1[:, 1], c="purple")
ax.scatter(X_2[:, 0], X_2[:, 1], c= "orange")
ax.set_ylabel("Feature #1")
ax.set_xlabel("Feature #0")
ax.set_title(f"Transformed K_means Seprate, r = {r}")


# In[2]:


## Question 3 code: 
class DimensionReduction:
    
    """
    How to run code: 
    
    > test sample data: 
    from sklearn.datasets import make_moons
    X, y = make_moons(20, noise=.05, random_state=0)
    
    > training: 
    1. dr = DimensionReduction(d = 2)
    -- d: integer, output dimension;
    
    2. dr.fit(X)
    
    3. dr.create_path_v2(num_neighbors = 3)
    -- num_neighbors: integer
    
    4. dr.train(echos = 10)
    -- echos: integer
    
    
    
    """
    
    
    def __init__(self,d, initvalue = 10, learningrate = 0.01):
        self.d = d
        self.initvalue = initvalue
        self.learningrate = learningrate
        self.echos = 0
        
    def knn(self, r):
        # Adj_mat = squareform(pdist(X, metric='euclidean'))
        Adj_mat = []
        for x in self.inputX:
            d = (self.inputX - x)**2
            distance = 0
            for i in range(self.inputdim):
                distance += d[:,i]
            Adj_mat.append(distance)
        self.Adj_mat = np.array(Adj_mat)
        
        
        W = np.zeros(self.Adj_mat.shape)
        
        Adj_sort = np.argsort(self.Adj_mat, axis=1)
        # Set the weight (i,j) to 1 when either i or j is within the k-nearest neighbors of each other
        for i in range(Adj_sort.shape[0]):
            W[i,Adj_sort[i,:][:(r+1)]] = 1
        
        self.W = W
        
        self.sparse_matrix = np.multiply(self.W, self.Adj_mat)
        return self
    
    def create_path_v2(self, num_neighbors):
        self.knn(num_neighbors)
        
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        
        self.graph_v2 = csr_matrix(self.sparse_matrix)
        Pi_v2 = []
        for i in range(self.inputsize):
            dist = dijkstra(csgraph=self.graph_v2, directed=False, indices=i, return_predecessors=False)
            Pi_v2.append(dist)
        
        self.Pi = np.array(Pi_v2)
        
        return self
        
    def calculate_y_distance(self):
        y_distance = []
        for y in self.y:
            d = (self.y - y)**2
            distance = 0
            for i in range(self.d):
                distance += d[:,i]
            y_distance.append(np.sqrt(distance))
        self.y_distance = np.array(y_distance)
        
        return self
        
    def get_derivative(self):
        self.dev = {}
        for i in range(self.inputsize):
            self.dev["y"+str(i)] = 0
            for j in range(self.inputsize):
                if self.y_distance[i][j] != 0:
                    coef = (self.y_distance[i][j] - self.Pi[i][j])/self.y_distance[i][j]
                    self.dev["y"+str(i)] += 2*coef*(self.y[i] - self.y[j])
                
        return self
    
    def update_y(self):
        for i in range(self.inputsize):
            self.y[i] -= self.learningrate*self.dev["y"+str(i)]
            
    def cost(self):
        cost = 0 
        for i in range(self.inputsize):
            for j in range(self.inputsize):
                cost += (self.y_distance[i][j] - self.Pi[i][j])**2
        
        return cost
        
    def fit(self, X):
        self.inputX = X
        self.inputsize = len(X)
        self.inputdim = len(X[0])
        
        # initialize yi for i = 1...n
        self.y = np.random.rand(self.inputsize, self.d)*self.initvalue
        
        return self
    
    def train(self, echos = 100):
        self.calculate_y_distance()
        cost = self.cost()
        print(f"The cost of initialization is {cost}.")
        for e in range(echos):
            self.get_derivative()
            self.update_y()
            self.calculate_y_distance()
            cost = self.cost()
            self.echos = self.echos+1
            print(f" echo {self.echos} cost: {cost}.")


# In[4]:


## test example make_moons:
from sklearn.datasets import make_moons
X, y = make_moons(20, noise=.05, random_state=0)
import numpy as np
np.random.seed(17) #0: 7.33 # 17: 3.2
dr = DimensionReduction(d = 2)
dr.fit(X)
dr.create_path_v2(num_neighbors = 3)
dr.train(echos = 10)


# In[ ]:


## run the code for swiss_roll: 
import pandas as pd
import numpy as np
df1 = pd.read_fwf('swiss_roll.txt', header =None)
X_1 = np.array(df1)
np.random.seed(17) #0: 7.33 # 17: 3.2
dr = DimensionReduction2(d = 2, learningrate=0.0001)
dr.fit(X_1)
dr.create_path_v2(num_neighbors = 50)
dr.train(echos = 100)


# In[ ]:


## run the code for swiss_roll_hole: 
import pandas as pd
import numpy as np
df2 = pd.read_fwf('swiss_roll_hole.txt', header =None)
X_2 = np.array(df1)
np.random.seed(17) #0: 7.33 # 17: 3.2
dr2 = DimensionReduction2(d = 2, learningrate=0.0001)
dr2.fit(X_2)
dr2.create_path_v2(num_neighbors = 50)
dr2.train(echos = 100)


# In[ ]:


# Plot the original data and using pca for swiss roll: 
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
ax.scatter(
    sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], s=50, alpha=0.8
)
ax.set_title("Swiss Roll in Ambient Space")
ax.view_init(azim=-66, elev=12)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
y_pca1 = pca.fit_transform(sr_points)

import matplotlib.pyplot as plt
_, (ax) = plt.subplots(sharex=True, sharey=True, figsize=(6, 4))

ax.scatter(y_pca1[:, 0], y_pca1[:, 1])
ax.set_ylabel("Feature #1")
ax.set_xlabel("Feature #0")
ax.set_title("Swiss Roll - PCA")


# In[ ]:


# Plot the original data and using pca for swiss roll hole: 
import pandas as pd
import numpy as np
df2 = pd.read_fwf('swiss_roll_hole.txt', header =None)
sr2_points = np.array(df2)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
ax.scatter(
    sr2_points[:, 0], sr2_points[:, 1], sr2_points[:, 2], s=50, alpha=0.8
)
ax.set_title("Swiss Roll in Ambient Space")
ax.view_init(azim=-66, elev=12)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
y_pca2 = pca.fit_transform(sr2_points)
import matplotlib.pyplot as plt
_, (ax) = plt.subplots(sharex=True, sharey=True, figsize=(6, 4))

ax.scatter(y_pca2[:, 0], y_pca2[:, 1])
ax.set_ylabel("Feature #1")
ax.set_xlabel("Feature #0")
ax.set_title("Swiss Roll Hole- PCA")


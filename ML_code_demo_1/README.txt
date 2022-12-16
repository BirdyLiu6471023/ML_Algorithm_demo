## Reference: 

Topic 2 reading notes: NN and kd-trees, And introductory tutorial on kd-tree;

## How to run my code: 

## 1. File Input:

1) If input is .txt file you would run it from beginning, transforming it into Euclidean space; I transform the Euclidean space into a data frame and save it into a .csv file which prevent me to do stemming and transforming again and again. So, if you have a.txt file, my code would transform your .txt file into a .csv file first;

2) If input is .csv file or .xlsx file with Euclidian space, please go directly to read_csv step. 

After running two class: 

## 2. Na•ve Bayes Code running instructions:
>>nb = NaiveBayes()       #no output
>>nb.fit(X_train,y_train) #not output, fit data
>>nb.pred(X_test)         #no output, predict data
>>nb.y_pred               #output our prediction
>>nb.acc(y_test)          #output accuracy

*`nb.acc()` would provide you the accuracy score. 
* function `compareNB` is to give the accuracy score for different training size in Naive Bayes Model. 
* paramter `k` is the smoother to avoid probability = 0.  

## 3. KNN code running instructions: 
>>nn = NearestNeighbour(depth) #defalt `depth =5`
>>nn.fit(X_train,y_train)
>>nn.pred(X_test, nums_of_nn) #defalt `nums_of_nn = 1` #no output predict data
>>nn.prediction               #output our prediction
>>nn.acc(y_test)              #output accuracy

*Before using the KNN, it is necessary to reduce the dimension of the data. So, I write a lot about how to decrease the dimension in analysis part. In conclusion, it is to cut the part that with low variance, only retaining around 2500 words in dataset and then using the KNN method to find the best cut. More details in analysis part

* due to time limit, I only use the depth=5 and I find the KNN (nums_of_nn=3) with the best column group described in ANALYSIS PART.



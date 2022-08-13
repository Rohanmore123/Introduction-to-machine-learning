#!/usr/bin/env python
# coding: utf-8

# #**EE769 Introduction to Machine Learning**
# 
# #Assignment 1: Gradient Descent, Linear Regression, and Regularization
# 
# 
# **Template and Instructions**
# 
# 
# 
# 1. Up to two people can team up, but only one should submit, and both should understand the entire code.
# 2. Every line of code should end in a comment explaining the line
# 3. It is recommended to solve the assignment in Google Colab.
# Write your roll no.s separated by commas here: 213230013, 213230007 .
# 4. Write your names here: Rohan Appaso More, Mirza Aman Baig.
# 5. There are two parts to the assignment. In the Part 1, the code format has to be strictly followed to enable auto-grading. In the second part, you can be creative.
# 6. **You can discuss with other groups or refer to the internet without being penalized, but you cannot copy their code and modify it. Write every line of code and comment on your own.**
# 
# 

# #**Part 1 begins ...**
# **Instructions to be strictly followed:**
# 
# 1. Do not add any code cells or markdown cells until the end of this part. Especially, do not change the blocks that say "TEST CASES, DO NOT CHANGE"
# 2. In all other cells only add code where it says "CODE HERE".
# 3. If you encounter any raise NotImplementedError() calls you may comment them out.
# 
# We cannot ensure correct grading if you change anything else, and you may be penalised for not following these instructions.

# ## Import Statements

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# ## Normalize function 
# 
# Add your code in the cell below to normalize the independent variables, making them zero mean and unit variance.

# In[2]:


def Normalize(X): # Output should be a normalized data matrix of the same dimension
    '''
    Normalize all columns of X using mean and standard deviation
    '''
    X_normalise = (X-np.mean(X,axis=0))/(np.std(X,axis=0)) #Normalising the data such that it has zero mean and unit variance
    return X_normalise

#     raise NotImplementedError()


# In[3]:


'''
TEST CASES, DO NOT CHANGE
'''
''' case 1 - 1 dimensional array'''
#X=np.array([[1,2,3],[3,4,5],[7,8,9]])
X1=np.array([1,2,3])
np.testing.assert_array_almost_equal(Normalize(X1),np.array([-1.224,  0.      ,  1.224]),decimal=3)
''' case 2 - 2 dimensional array'''
X2=np.array([[4,7,6],[3,8,9],[5,11,10]])
np.testing.assert_array_almost_equal(Normalize(X2),np.array([[ 0.  , -0.980581, -1.372813],[-1.224745, -0.392232,  0.392232],[ 1.224745,  1.372813,  0.980581]]))
''' case 3 - 1 dimensional array with float'''
X3=np.array([5.5,6.7,3.2,6.7])
np.testing.assert_array_almost_equal(Normalize(X3),np.array([-0.017,  0.822, -1.627,  0.822]),decimal=3)


# ## Prediction Function
# 
# Given X and w, compute the predicted output. Do not forget to add 1's in X

# In[4]:


def Prediction (X, w): # Output should be a prediction vector y
    '''
    Compute Prediction given an input datamatrix X and weight vecor w. Output y = [X 1]w where 1 is a vector of all 1s 
    '''
    n,m = X.shape  #storing number of rows and columns of X in n and m correspondingly.
    X0 = np.ones((n,1)) # creating a vector of ones.
    Xnew = np.hstack((X,X0))  #Now finally we can concatenate the X0 vector 
    y = np.matmul(Xnew,w)   # our prediction function is y which is y = Xnew*w
    return y
# #   
    #raise NotImplementedError()


# In[5]:


'''
TEST CASES, DO NOT CHANGE
'''
''' case 1 - Known input output matrix and weights 1'''
X1 = np.array([[3,2],[1,1]])
w1 = np.array([2,1,1]) 
np.testing.assert_array_equal(Prediction(X1,w1),np.array([9,4]))


# ## Loss Functions
# 
# Code the four  loss functions:
# 
# 1. MSE loss is only for the error
# 2. MAE loss is only for the error
# 3. L2 loss is for MSE and L2 regularization, and can call MSE loss
# 4. L1 loss is for MSE and L1 regularization, and can call MSE loss

# In[6]:


def MSE_Loss (X, t, w, lamda =0): # Ouput should be a single number
    '''
    lamda=0 is a default argument to prevent errors if you pass lamda to a function that doesn't need it by mistake. 
    This allows us to call all loss functions with the same input format.
    
    You are encouraged read about default arguments by yourself online if you're not familiar.
    '''
    n,m = X.shape 
    X0 = np.ones((n,1))
    Xnew = np.hstack((X,X0))
    y = np.matmul(Xnew,w)

    mse = (np.square(t-y)).mean() #using the standard formula of mse
    return mse
        
 
    raise NotImplementedError()


# In[7]:


'''
TEST CASES, DO NOT CHANGE
'''
''' case 1 '''
X=np.array([[3,6,5],[4.5,6.6,6]])
t=np.array([4,5.5])
w=np.array([2,-1,0.5,1])
np.testing.assert_almost_equal(MSE_Loss(X,t,w),0.53,decimal=3)


# In[8]:


def MAE_Loss (X, t, w, lamda = 0): # Output should be a single number
    n,m = X.shape 
    X0 = np.ones((n,1))
    Xnew = np.hstack((X,X0))
    y = np.matmul(Xnew,w)

    k2 = np.absolute((t-y)) #using the standard formula of MAE
    mae = k2.mean()
    return mae
    
    
    # YOUR CODE HERE
    #raise NotImplementedError()


# In[9]:


'''
TEST CASES, DO NOT CHANGE
'''
''' case 1 '''
X=np.array([[3,6,5],[4.5,6.6,6]])
t=np.array([4,5.5])
w=np.array([2,-1,0.5,1])
np.testing.assert_almost_equal(MAE_Loss(X,t,w),0.700,decimal=3)


# In[10]:


def L2_Loss (X, t, w, lamda=0): # Output should be a single number
    ''' Need to specify what inputs are'''
    n,m = X.shape 
    X0 = np.ones((n,1))
    l = 0.5  # l is lambda
    Xnew = np.hstack((X,X0))
    y = np.matmul(Xnew,w)
    term1 = (np.square(t-y)).mean() #this is mse which is first term of L2
    p = w.shape 
    q = p[0]  
    
    wnew = np.delete(w,q-1)  #dropping the bias
    sqr_wnew = np.square(wnew)
    sum_sqr_wnew = np.sum(sqr_wnew) #using norm for second term
    term2 = l*(np.sqrt(sum_sqr_wnew))
    final = term1+term2
    return final
    
    # YOUR CODE HERE
    
    #raise NotImplementedError()


# In[11]:


'''
TEST CASES, DO NOT CHANGE
'''
''' case 1 '''
X=np.array([[3,6,5],[4.5,6.6,6]])
t=np.array([4,5.5])
w=np.array([2,-1,0.5,1])
np.testing.assert_almost_equal(L2_Loss(X,t,w,0.5),1.675,decimal=3)


# In[12]:


def L1_Loss (X, t, w, lamda): # Output should be a single number
    p = w.shape
    q = p[0]
    wnew = np.delete(w,q-1)
    term2 = lamda*(np.sum(np.absolute(wnew)))  #for second term of L1 using standard formula
    final = MSE_Loss (X, t, w, lamda =0) + term2
    return final
    # YOUR CODE HERE
    raise NotImplementedError()


# In[13]:


'''
TEST CASES, DO NOT CHANGE
'''
''' case 1 '''
X=np.array([[3,6,5],[4.5,6.6,6]])
t=np.array([4,5.5])
w=np.array([2,-1,0.5,1])
np.testing.assert_almost_equal(L1_Loss(X,t,w,0.5),2.280,decimal=3)


# In[14]:


def NRMSE_Loss (X, t, w): # Output should be a single number
    numr = np.sqrt(MSE_Loss (X, t, w, lamda =0)) #calling the mse function defined above and taking its square root to get RMSE
    denr = t.std() 
    nrmse = numr/denr     #NRMSE_Loss = RMSE/standard deviation of t. where t is the actual value of output.
    return nrmse
    
    # YOUR CODE HERE
    raise NotImplementedError()


# In[15]:


'''
TEST CASES, DO NOT CHANGE
'''
''' Test case 1 '''
X=np.array([[3,6,5],[4.5,6.6,6]])
t=np.array([4,5.5])
w=np.array([2,-1,0.5,1])
np.testing.assert_almost_equal(NRMSE_Loss(X,t,w),0.970,decimal=3)


# ## Gradient function
# Each Loss function will have its own gradient function:
# 
# 1. MSE gradient is only for the error
# 2. MAE gradient is only for the error
# 3. L2 gradient is for MSE and L2 regularization, and can call MSE gradient
# 4. L1 gradient is for MSE and L1 regularization, and can call MSE gradient

# In[16]:


def MSE_Gradient (X, t, w, lamda=0):
    n,m = X.shape 
    X0 = np.ones((n,1))
    l = 0.5 #l is lambda
    Xnew = np.hstack((X,X0))
    y = np.matmul(Xnew,w) #y is predicted value of output.
    errr = t-y
    final_mse_gr = -1*(np.matmul(errr,Xnew)) #using mse gradient formula
    return final_mse_gr
    
    # YOUR CODE HERE
    raise NotImplementedError()


# In[17]:


'''
TEST CASES, DO NOT CHANGE
'''
''' case 1 '''
X=np.array([[3,6,5],[4.5,6.6,6]])
t=np.array([4,5.5])
w=np.array([2,-1,0.5,1])
np.testing.assert_array_almost_equal(MSE_Gradient(X,t,w),np.array([2.55, 2.94, 2.9 , 0.4 ]),decimal=3)


# In[18]:


def MAE_Gradient (X, t, w, lamda=0): # Output should have the same size as w
    n,m = X.shape 
    X0 = np.ones((n,1))
    l = 0.5  # l is lambda
    Xnew = np.hstack((X,X0))
    y = np.matmul(Xnew,w)
    i1 = np.sign(t-y)  #using sign function to assign sign to error
    i2 = -0.5*(np.matmul(i1,Xnew))  #using the formula of MAE Gradient
    return i2
    # YOUR CODE HERE
    raise NotImplementedError()


# In[19]:


'''
TEST CASES, DO NOT CHANGE
'''
''' case 1 '''
X=np.array([[3,6,5],[4.5,6.6,6]])
t=np.array([4,5.5])
w=np.array([2,-1,0.5,1])
np.testing.assert_array_almost_equal(MAE_Gradient(X,t,w),np.array([0.75,  0.3 ,  0.5 , 0.]),decimal=3)


# In[20]:


def L2_Gradient (X, t, w, lamda): # Output should have the same size as w
    n,m = X.shape 
    X0 = np.ones((n,1))
    l = 0.5  # l is lambda
    Xnew = np.hstack((X,X0))
    y = np.matmul(Xnew,w)
    error = t-y
    fin_mse_gr = -1*(np.matmul(error,Xnew)) #calulating mse gradient which is the first term of L2_gradient
    p = w.size
    w1 = w
    w1[p-1] = 0
    norm_w = np.sqrt(np.sum(np.square(w1))) # taking norm which is second term.
    tr2 = (lamda/norm_w)*w1
    fnl5 = fin_mse_gr + tr2 #Now adding two terms to get the L2 Gradient.
    return fnl5
    
    # YOUR CODE HERE
    raise NotImplementedError()


# In[21]:


'''
TEST CASES, DO NOT CHANGE
'''
''' case 1 '''
X=np.array([[3,6,5],[4.5,6.6,6]])
t=np.array([4,5.5])
w=np.array([2,-1,0.5,1])
np.testing.assert_array_almost_equal(L2_Gradient(X,t,w,0.5),np.array([2.986, 2.721, 3.009 , 0.4 ]),decimal=3)


# In[22]:


def L1_Gradient (X, t, w, lamda): # Output should have the same size as w
    n,m = X.shape 
    X0 = np.ones((n,1))
    l = 0.5   #l is lambda
    Xnew = np.hstack((X,X0))
    y = np.matmul(Xnew,w)  #predicted value of output
    error = t-y
    fin_mse_gr = -1*(np.matmul(error,Xnew))  # using formula to calculate mse which first term of L1 gradient.
    w1 = w
    w1[3] = 0  #assigning last column of w to zero to maintain dimensions.
    term2 = l*(np.sign(w1))  #Now multiplying with lambda to the sign of weights
    finnnal = fin_mse_gr + term2   #now adding both terms to get L1 Gradient
    return finnnal
    
    
    # YOUR CODE HERE
    raise NotImplementedError()


# In[23]:


'''
TEST CASES, DO NOT CHANGE
'''
''' case 1 '''
X=np.array([[3,6,5],[4.5,6.6,6]])
t=np.array([4,5.5])
w=np.array([2,-1,0.5,1])
np.testing.assert_array_almost_equal(L1_Gradient(X,t,w,0.5),np.array([3.05, 2.44, 3.4 , 0.4 ]),decimal=3)


# ## Gradient Descent Function
# 

# In[24]:


def Gradient_Descent (X, X_val, t, t_val, w, lamda, max_iter, epsilon, lr, lossfunc, gradfunc): # See output format in 'return' statement
    
    # YOUR CODE HERE
    raise NotImplementedError()
    return w_final, train_loss_final, validation_loss_final, validation_NRMSE #You should return variables structured like this


# In[25]:


'''
TEST CASES, DO NOT CHANGE
'''
np.random.seed(2)

X=np.array([[23,24],[1,2]])
t=np.array([4,5])
X_val=np.array([[3,4],[5,6]])
t_val=np.array([3,4])
w=np.array([3,2,1])
results =Gradient_Descent (X, X_val, t, t_val, w, 0.1, 100, 1e-10, 1e-5, L2_Loss,L2_Gradient) 
np.testing.assert_array_almost_equal([results[1],results[2]],[697.919,17.512],decimal=3)


# ## Pseudo Inverse Method
# 
# You have to implement a slightly more advanced version, with L2 penalty:
# 
# w = (X' X + lambda I)^(-1) X' t.
# 
# See, for example: Section 2 of https://web.mit.edu/zoya/www/linearRegression.pdf

# In[34]:


def Pseudo_Inverse (X, t, lamda): # Output should be weight vector
    f0 = np.array([[1],[1]])
    xnew = np.concatenate([X,f0],axis=1) #making last column of X as 1.
    f = np.matmul((xnew.T),xnew)  #Now multiplying both 
    f1 = f+ lamda* (np.identity(4))  #using formula of pseudo inverse
    f2 = (np.linalg.inv(f1))
    f3 = np.matmul(f2,(xnew.T))
    f5 = np.matmul(f3,t)
    return f5

    # YOUR CODE HERE
    raise NotImplementedError()


# In[35]:


'''
TEST CASES, DO NOT CHANGE
'''
''' case 1 - other data'''
X=np.array([[3,6,5],[4.5,6.6,6]])
t=np.array([4,5.5])
np.testing.assert_array_almost_equal(Pseudo_Inverse(X,t,0.5),np.array([ 0.491,  0.183,  0.319, -0.002]),decimal=3)


# #Save the code above this as a RollNo1_RollNo2_1.py file after running the test blocks to make sure there are no errors.

# #**... Part 1 ends**
# Below this you be more creative. Just comment out the lines where you save files (e.g. test predictions).

# #**Part 2 begins ...**
# 
# **Instructions to be loosely followed (except number 8):**
# 
# 
# 1. Add more code and text cells between this and the last cell.
# 2. Read training data from: https://www.ee.iitb.ac.in/~asethi/Dump/TempTrain.csv only. Do not use a local copy of the dataset.
# 3. Find the best lamda for **MSE+lamda*L2(w)** loss function. Plot training and validation RMSE vs. 1/lamda (1/lamda represents model complexity). Print weights, validation RMSE, validation NMSE for the best lamda.
# 4. Find the best lamda for **MSE+lamda*L1(w)** loss function. Plot training and validation RMSE vs. 1/lamda (1/lamda represents model complexity). Print weights, validation RMSE, validation NMSE for the best lamda.
# 5. Find the best lamda for the **pseudo-inv method**. Plot training and validation RMSE vs. 1/lamda (1/lamda represents model complexity). Print weights, validation RMSE, validation NMSE for the best lamda.
# 6. Write your observations and conclusions.
# 7. Read test data from: https://www.ee.iitb.ac.in/~asethi/Dump/TempTest.csv only. Do not use a local copy of the dataset. Predict its dependent (missing last column) using the model with the lowest MSE, RMSE, or NMSE. Save it as a file RollNo1_RollNo2_1.csv.
# 8. **Disable the prediction csv file saving statement and submit this entire .ipynb file (part 1 and part 2), .py file (part 1 only), and .csv file as a single RollNo1_RollNo2_1.zip file.**
# 

# #**... Part 2 ends.**
# 
# 1. Write the name or roll no.s of friends from outside your group with whom you discussed the assignment here (no penalty for mere discussion without copying code): 
# 2. Write the links of sources on the internet referred here (no penalty for mere consultation without copying code): 

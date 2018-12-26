
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import utils
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error


# In[35]:


input=pd.read_csv('/Users/blair/Desktop/RFRegressor/input_386.csv')#Put address
efp=pd.read_csv('/Users/blair/Desktop/RFRegressor/ecfp_256.csv')
efp=np.array(efp)
Isomer=input[['Isomer']].copy()
X=np.concatenate((efp,Isomer),axis=1)


# In[36]:


X=preprocessing.normalize(X)
#print(X)
#print(len(X))


# In[37]:


Y=input[['Hammett Constant']].copy()
Y=np.array(Y)
#print(y)
#print(len(Y))
#print(type(Y))
#print(Y)


# In[38]:


def bootstrap_resample(X, n=None):
   
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    i= resample_i
    return i


# In[39]:


MSE_boot=np.zeros((100),dtype= float)
Index= np.zeros((100,386),dtype=int)
Y_prediction=np.zeros(386).T
Y_pred_all=np.zeros((386,100),dtype=float)


# In[40]:


regr = RandomForestRegressor(n_estimators=100, max_depth=None,
                             random_state=0,bootstrap=0)


# In[41]:


for m in range(0,100):
    ind = bootstrap_resample(X,n=386) #Call bootstrap to randomly select index
    Index[m]=ind
    X_resample= X[ind]
    Y_resample= Y[ind]
    X_testbsr=np.delete(X,ind,axis=0)
    Y_testbsr=np.delete(Y,ind,axis=0)
    y2=[]
    for t in Y_resample:
        y2.append(float(str(t)[1:-1]))
    Y_resample=y2
    #print(Y_resample)
    regr.fit(X_resample, Y_resample)
    Y_pred_resample=regr.predict(X_testbsr)  #Fit ANN on test
    Y_pred_train=regr.predict(X_resample)    #Fit ANN on training set
    Y_pred_all[:,m]=regr.predict(X)          # Fit ANN on complete database
    MSE_train=mean_squared_error(Y_resample, Y_pred_train)
    MSE_test= mean_squared_error(Y_testbsr, Y_pred_resample)
    MSE_boot[m]=(MSE_train*0.368)+(0.632)*(MSE_test)


# In[42]:


#print(utils.multiclass.type_of_target(Y.astype('int')))


# In[43]:


z=min(MSE_boot)
print(z)
d=np.sort(Y_pred_all)
alpha=0.95
lower=np.zeros(386)
upper=np.zeros(386)
mean=np.mean(d,axis=1)
p = ((1.0-alpha)/2.0) * 100
q = (alpha+((1.0-alpha)/2.0)) * 100
for j in range(0,386):
    min_=min(d[j])
    max_=max(d[j])
    lower[j] = max(min_, np.percentile(d[j,:], p)) #Lower limit
    upper[j]=min(max_, np.percentile(d[j,:], q))   #Upper limit


# In[44]:


pd.DataFrame(lower).to_csv('/Users/blair/Desktop/RFRegressor/lowerlimit.csv') #Put address
pd.DataFrame(upper).to_csv('/Users/blair/Desktop/RFRegressor/upperlimit.csv')#Put address
pd.DataFrame(mean).to_csv('/Users/blair/Desktop/RFRegressor/mean.csv')#Put address
pd.DataFrame(MSE_boot).to_csv('/Users/blair/Desktop/RFRegressor/mse.csv')#Put address
b=np.argmin(MSE_boot)
min_index=Index[b]
min_index=np.array(min_index)
train_best= X[min_index]
train_best.shape
Y_best_train= Y[min_index]
#Predicting output for model with least error
y3=[]
for t in Y_best_train:
    y3.append(float(str(t)[1:-1]))
Y_best_train=y3
regr.fit(train_best,Y_best_train)
Y_prediction=regr.predict(X)
#Save output in csv
#pd.DataFrame(Y_test).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/tanya/Y_test.csv')#Put address
pd.DataFrame(Y_prediction).to_csv('/Users/blair/Desktop/RFRegressor/Y_prediction.csv')#Put address
#pd.DataFrame(Y_target).to_csv('/Users/blair/Desktop/RFRegressor/Y_target.csv') #Put address


# In[45]:


X_best_testbsr=np.delete(X,min_index,axis=0)
Y_best_testbsr=np.delete(Y,min_index,axis=0)
Y_pred_best_testbsr=regr.predict(X_best_testbsr)
pd.DataFrame(Y_pred_best_testbsr).to_csv('/Users/blair/Desktop/RFRegressor/Y_best_pred_testbsr.csv')
pd.DataFrame(Y_best_testbsr).to_csv('/Users/blair/Desktop/RFRegressor/Y_best_testbsr.csv')


# In[46]:


depth=regr.get_params(deep=1)
print(depth)


# In[27]:


lab_enc = preprocessing.LabelEncoder()
Target = lab_enc.fit_transform(Y)
print(type(Y))
#print(y)
print(utils.multiclass.type_of_target(Y))


# In[219]:


list(lab_enc.inverse_transform_transform(y))


# In[ ]:


clf.fit(predictiondata, Target)


# In[ ]:


c=predictiondata[-1]
print(utils.multiclass.type_of_target(c))
#c=array.reshape(1, -1)
print(c)
print(type(c))
print(len(c))


# In[ ]:


print(clf.predict([c]))


# In[ ]:


list(lab_enc.inverse_transform_transform(75))


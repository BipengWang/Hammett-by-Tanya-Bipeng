
# In[2]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import utils
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error


# In[337]:


input=pd.read_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/rf/input_386.csv')#Put address
efp=pd.read_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/rf/ecfp.csv')
efp=np.array(efp)
Isomer=input[['Isomer']].copy()
X=np.concatenate((efp,Isomer),axis=1)


# In[338]:


X=preprocessing.normalize(X)
#print(X)
#print(len(X))


# In[339]:


Y=input[['Hammett Constant']].copy()
Y=np.array(Y)
#print(y)
#print(len(Y))
#print(type(Y))
#print(Y)


# In[1]:


y0=[]
for m in Y:
    if m < 0:
        m = -1
    else:
        if m == 0:
            m = 0
        else:
            m = 1   
    y0.append(m)

Y_target=y0
Y_target=np.array(Y_target)
pd.DataFrame(Y_target).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/rf/Y_target.csv') #Put address
#print(Y_target)


# In[341]:


y1=[]
for m in Y:
    if m < 0:
        m = [-1]
    else:
        if m == 0:
            m = [0]
        else:
            m = [1]      
    y1.append(m)

Y=y1
Y=np.array(Y)

#print(type(Y))
#print(Y)


# In[342]:


def bootstrap_resample(X, n=None):
   
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    i= resample_i
    return i


# In[343]:


MSE_boot=np.zeros((10000),dtype= float)
Index= np.zeros((10000,386),dtype=int)
Y_prediction=np.zeros(386).T
Y_pred_all=np.zeros((386,10000),dtype=float)


# In[344]:


clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             random_state=0,bootstrap=0)


# In[345]:


for m in range(0,10000):
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
    clf.fit(X_resample, Y_resample)
    Y_pred_resample=clf.predict(X_testbsr)  #Fit ANN on test
    Y_pred_train=clf.predict(X_resample)    #Fit ANN on training set
    Y_pred_all[:,m]=clf.predict(X)          # Fit ANN on complete database
    MSE_train=mean_squared_error(Y_resample, Y_pred_train)
    MSE_test= mean_squared_error(Y_testbsr, Y_pred_resample)
    MSE_boot[m]=(MSE_train*0.368)+(0.632)*(MSE_test)


# In[346]:


#print(utils.multiclass.type_of_target(Y.astype('int')))


# In[347]:


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


# In[348]:


pd.DataFrame(lower).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/rflowerlimit.csv') #Put address
pd.DataFrame(upper).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/rf/upperlimit.csv')#Put address
pd.DataFrame(mean).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/rf/mean.csv')#Put address
pd.DataFrame(MSE_boot).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/rf/mse.csv')#Put address
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
clf.fit(train_best,Y_best_train)
Y_prediction=clf.predict(X)
#Save output in csv
#pd.DataFrame(Y_test).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/tanya/Y_test.csv')#Put address
pd.DataFrame(Y_prediction).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/rf/Y_prediction.csv')#Put address


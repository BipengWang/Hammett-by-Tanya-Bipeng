import pandas as pd
import numpy as np
input=pd.read_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/tanya/input.csv')#Put address
# Define output Y
Y=input[['Hammett Constant']].copy()
# Define variables for isomer type and dipole moments from input
Isomer=input[['Isomer']].copy()
efp=pd.read_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/tanya/EFP.csv')#Put address
efp=np.array(efp)
X=np.concatenate((efp,Isomer),axis=1)
from sklearn import preprocessing
X=preprocessing.normalize(X)
Y=np.array(Y)
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
def bootstrap_resample(X, n=None):
   
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    i= resample_i
    return i
#Define null arrays
MSE_boot=np.zeros((10000),dtype= float)
Index= np.zeros((10000,395),dtype=int)
#Store bootstrap outputs
Y_prediction=np.zeros(395).T
Y_pred_all=np.zeros((395,10000),dtype=float)
#Define ANN model with 250 hidden neurons and alpha =0.05
mlp_resample = MLPRegressor(hidden_layer_sizes=(250),alpha = 0.05)
#Bootstrap resampling

for m in range(0,10000):
    ind = bootstrap_resample(X,n=395) #Call bootstrap to randomly select index
    Index[m]=ind
    X_resample= X[ind]
    Y_resample= Y[ind]
    X_testbsr=np.delete(X,ind,axis=0)
    Y_testbsr=np.delete(Y,ind,axis=0)
    #ANN
    mlp_resample.fit(X_resample,Y_resample)
    Y_pred_resample=mlp_resample.predict(X_testbsr)  #Fit ANN on test
    Y_pred_train=mlp_resample.predict(X_resample)    #Fit ANN on training set
    Y_pred_all[:,m]=mlp_resample.predict(X)          # Fit ANN on complete database
    MSE_train=mean_squared_error(Y_resample, Y_pred_train)
    MSE_test= mean_squared_error(Y_testbsr, Y_pred_resample)
    MSE_boot[m]=(MSE_train*0.368)+(0.632)*(MSE_test)

z=min(MSE_boot)
print(z) 
#print(Y_pred_all[:,2])
#Confidence intervals
d=np.sort(Y_pred_all)
alpha=0.95
lower=np.zeros(395)
upper=np.zeros(395)
mean=np.mean(d,axis=1)
p = ((1.0-alpha)/2.0) * 100
q = (alpha+((1.0-alpha)/2.0)) * 100
for j in range(0,395):
    min_=min(d[j])
    max_=max(d[j])
    lower[j] = max(min_, np.percentile(d[j,:], p)) #Lower limit
    upper[j]=min(max_, np.percentile(d[j,:], q))   #Upper limit

pd.DataFrame(lower).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/tanya/lowerlimit_10000.csv') #Put address
pd.DataFrame(upper).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/tanya/upperlimit_10000.csv')#Put address
pd.DataFrame(mean).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/tanya/means_10000.csv')#Put address
pd.DataFrame(MSE_boot).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/tanya/mse_10000.csv')#Put address
b=np.argmin(MSE_boot)
min_index=Index[b]
min_index=np.array(min_index)
train_best= X[min_index]
train_best.shape
Y_best_train= Y[min_index]
#Predicting output for model with least error
mlp_resample.fit(train_best,Y_best_train)
Y_prediction=mlp_resample.predict(X)
#Save output in csv
#pd.DataFrame(Y_test).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/tanya/Y_test.csv')#Put address
pd.DataFrame(Y_prediction).to_csv('/home/rcf-proj/sm1/bipengwa/project/pymatgen/tanya/Y_prediction_10000.csv')#Put address



    
    

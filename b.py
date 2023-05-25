
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import operator
import pandas as pd
lg=LinearRegression()
import pickle
trainfile = open("./data/train.pkl", 'rb')
testfile = open("./data/test.pkl", 'rb')
trainP = pickle.load(trainfile)
testP = pickle.load(testfile)
train=np.array(trainP)
test=np.array(testP)
np.random.shuffle(train)
np.random.shuffle(train)
trainset=[]
for i in range(10):
  y=i*800
  trainset.append(train[y:y+800,:])
  trainset=np.array(trainset)
  test_x=test[:,0]
test_y=test[:,1]
# print(test)
lrg=LinearRegression()
bias2=[]
mse=[]
biAs=[]
ind_list=[]
variance=[]
irr_err=[]
sort_axis = operator.itemgetter(0)
sorted_zip1 = sorted(zip(test_x,test_y), key=sort_axis)
test_x, test_y = zip(*sorted_zip1)
for j in range(1,21,1):
  ind_list.append(j)
  vari=0
  bias=0
  exp=0
  f_cap=[]
  var_arr=[]
  avd=[]
  for i in range(len(test[:,0])):
      f_cap.append(0)
      var_arr.append(0)
      # print(test_x[i],test_y[i])
  varian=0.0
  mse_ar=np.zeros(len(test_x))
  for i in range(10):
      lol=trainset[i]
      y = lol[:,1]
      poly_f= PolynomialFeatures(degree=j)
      x_poly = poly_f.fit_transform(lol[:,0].reshape(-1,1))
      reg=lrg.fit(x_poly,y)
      test_poly= poly_f.fit_transform(test[:,0].reshape(-1,1))
      predic=lrg.predict(test_poly)
      predic=np.array(predic)
      sorted_zip = sorted(zip(test[:,0],predic), key=sort_axis)
      test_x, predic = zip(*sorted_zip)
#       plt.scatter(test_x,test_y)
#       plt.plot(test_x,predic)
      avd.append(predic)
      # print(test.shape)
      for a in range(len(test_x)):
        f_cap[a]+=predic[a]
        mse_ar[a]+=(test_y[a]-predic[a])**2
        # print(test_y[a],test_x[a])
        # print(predic[a],test_x[a])
        # print('------------------')
      # plt.show()
  mse.append(np.average(mse_ar)/10)
  sorted_zip1 = sorted(zip(test_x,test_y), key=sort_axis)
  test_x, test_y = zip(*sorted_zip1)
  newBias=0
  for i in range(len(f_cap)):
      f_cap[i]= f_cap[i]/10
      # print(f_cap[i],test_y[i])
      bias+=(f_cap[i]-test_y[i])**2
      newBias+=abs(f_cap[i]-test_y[i])
      # print(bias)
#   plt.scatter(test_x,test_y)
#   plt.plot(test_x,f_cap,'blue')
#   plt.title('Graph for degree %s'%(j))
#   plt.show()
  for i in range(10):
    for x in range(len(test_x)):
      var_arr[x]+=(avd[i][x]-f_cap[x])**2
  for x in range(len(test_x)):
      var_arr[x]=var_arr[x]/10
  var_arr= np.array(var_arr)
  varian=np.average(var_arr)
  variance.append(varian)
  bias=bias/len(f_cap)
  newBias = newBias/len(f_cap)
  irr_err.append(np.average(mse_ar)/10-(bias+varian))
  bias2.append(bias)
  biAs.append(newBias)
plt.plot(ind_list,bias2)
plt.plot(ind_list,variance,'orange')
plt.plot(ind_list,mse,'pink')
plt.legend(["Bias sq.","Variance","MSE", "Irreducible Error"])
plt.ylabel('Error')
plt.xlabel('Model Complexity')
bv_df = pd.DataFrame(list(zip(ind_list,biAs,bias2,variance)))
bv_df.columns=['Complexity','Bias','Bias^2','Variance']
irr_df=pd.DataFrame(list(zip(ind_list,irr_err)))
irr_df.columns=['Complexity','Irreducible Error']
print('Table-1')
print(bv_df)
print('Table-2')
print(irr_df)
# print(bias2)
# print(variance)
plt.show()
plt.plot(ind_list,irr_err,'blue')
plt.plot([0,20],[0,0],'red')
plt.ylabel('Irreducible Error')
plt.xlabel('Model Complexity')
plt.show()
print(biAs)
print(mse)
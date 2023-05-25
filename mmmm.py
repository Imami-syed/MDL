lrg=LinearRegression()
bisasq=[0]*15
biAs=[0]*15
index=[i for i in range(1,16)]
variance=[0]*15
error=[0]*15
mse=[]
test_x=test[:,0]
test_y=test[:,1]
varii=[0]*15
biasss=[0]*15
# print(test)
test_x, test_y = zip(*sorted(zip(test_x,test_y), key=operator.itemgetter(0)))
for j in range(1,16):
  f_cap=[0]*len(test[:,0])
  var_arr=[0]*len(test[:,0])
  avd=[]
  varian=0.0
  mse_ar=[0]*len(test_x)
  for i in range(16):
      y = trainset[i][:,1]
      y0=trainset[i][:,0]
      poly_f= PolynomialFeatures(degree=j)
      x_poly = poly_f.fit_transform(y0.reshape(-1,1))
      reg=lrg.fit(x_poly,y)
      predic=lrg.predict(poly_f.fit_transform(test[:,0].reshape(-1,1)))
      test_x, predic = zip(*sorted(zip(test[:,0],predic), key=operator.itemgetter(0)))
      avd.append(predic)
      f_cap=np.add(f_cap,predic)
      mse_ar=np.add(mse_ar,np.square(np.subtract(predic,test_y)))
  mse.append(np.mean(mse_ar)/16)
  test_x, test_y = zip(*sorted(zip(test_x,test_y), key=operator.itemgetter(0)))
  newBias=0
  f_cap=np.divide(f_cap,16)
  bias=np.sum(np.square(np.subtract(f_cap,test_y)))
  newBias=np.sum(np.abs(np.subtract(f_cap,test_y)))
  for x in range(len(test_x)):
    for i in range(16):
      var_arr[x]+=(avd[i][x]-f_cap[x])**2
  var_arr=np.divide(var_arr,16)
  variance[j-1]=np.mean(var_arr)
  error[j-1]=np.mean(mse_ar)/16-((bias/len(f_cap))+np.mean(var_arr))
  bisasq[j-1]=bias/len(f_cap)
  biAs[j-1]=newBias/len(f_cap)



tab1 = pd.DataFrame(list(zip(index,biAs,bisasq,variance)))
tab1.columns=['Index','Bias','Bias sqaure','Variance']
tab2=pd.DataFrame(list(zip(index,error)))
tab2.columns=['Complexity','Irreducible Error']
print("Table-1 : ")
print(tab1)

print("Table-2 : ")
print(tab2)
plt.plot(index,bisasq)
plt.plot(index,variance)
plt.plot(index,mse)
plt.legend(["Bias square","Variance","M.S.E.", "Irreducible Error"])
plt.ylabel('Error on y-axis')
plt.xlabel('Complexity on x-axis')
plt.show()
plt.show()
plt.plot(index,error)
plt.plot([0,15],[0,0])
plt.ylabel('Error on y-axis')
plt.xlabel('Complexity on x-axis')

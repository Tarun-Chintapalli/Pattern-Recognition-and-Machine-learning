import numpy as np
import math
import matplotlib.pyplot as plt
train_dat = np.genfromtxt('Wage_dataset.csv', delimiter=',')
train_data = np.array(train_dat)
year =train_data[0:2250,0]
age = train_data[0:2250,1]
edu = train_data[0:2250,4]
wage= train_data[0:2250,10]
yeart =train_data[2250:3000,0]
aget = train_data[2250:3000,1]
edut = train_data[2250:3000,4]
waget= train_data[2250:3000,10]
error=0
degree=3 #input the degree here
Y=[]
i=0
x=np.linspace(2000,2020,400)
x=np.array(x)
X=np.ones(np.size(year))
X=np.transpose(X)
for i in range(degree):
	X=np.c_[X,year**(i+1)]
W=np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),wage))
Y=W[0]*np.ones(400)
for i in range(1,degree+1):
	Y=Y+W[i]*(x**i)
for i in range(0,750):
	a=0
	for j in range(0,degree+1):
		a=a+W[j]*(yeart[i]**j)
	error=error+(a-waget[i])**2
print(error)    
plt.xlabel("Wage")
plt.ylabel("Year")
plt.title("Polynomial regression for Year Vs Wage")           # error
plt.plot(year,wage,'r.')        # points
plt.plot(x,Y)                   # Plot of the curve
plt.show()

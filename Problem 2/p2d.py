import numpy as np
import math
import matplotlib.pyplot as plt
cf=0 ;cs=0 ;i=0 ;k=0;ms=0;mf=0;df=[];mx=[];my=[];ds=[];fin=[];lj=[];sx=[];sy=[];l=[];lf=0;ls=0;lcf=0;lcs=0;fs=0;ff=0;mxt=[];myt=[];sxt=[];syt=[]
train_label=[]
train_dat = np.genfromtxt('P2_train.csv', delimiter=',')
test_dat = np.genfromtxt('P2_test.csv', delimiter=',')
train_data=np.array(train_dat)
train_label = train_data[:,2]
train_data = train_data[:,(0,1)]
test_data=np.array(test_dat)
test_label = test_data[:,2]
test_data=test_data[:,(0,1)]
for i in range (0, 310):
	if(train_label[i]==1):
		mf=mf+train_data[i]
		mx.append(train_data[i,0])
		my.append(train_data[i,1])
		df.append(train_data[i])
		cf=cf+1
	else:
		ms=ms+train_data[i]
		sx.append(train_data[i,0])
		sy.append(train_data[i,1])
		ds.append(train_data[i])
		cs=cs+1
sf=np.cov(np.transpose(df))  
ss=np.cov(np.transpose(ds))
mf=mf/cf
ms=ms/cs
prf=float(cf)/float(cf+cs)
prs=float(cs)/float(cs+cf)
ocov=prf*sf+prs*ss
x = np.linspace(-10, 10 ,400)
y = np.linspace(-10, 10 ,400)
g1=np.array([(-math.log(2*math.pi) - (math.log(np.linalg.det(sf)))/2 - (np.matmul(np.transpose(np.array([i,j])-mf),np.matmul(np.linalg.inv(sf),(np.array([i,j])-mf))))/2 + math.log(prf))  for j in y for i in x ])
g2=np.array([(-math.log(2*math.pi) - (math.log(np.linalg.det(ss)))/2 - (np.matmul(np.transpose(np.array([i,j])-ms),np.matmul(np.linalg.inv(ss),(np.array([i,j])-ms))))/2 + math.log(prs))  for j in y for i in x ])

fn=(g1-g2)
F=fn.reshape(400,400)
G=g1.reshape(400,400)
H=g2.reshape(400,400)
for i in range (0, 90):
	if(test_label[i]==1):
		mxt.append(test_data[i,0])
		myt.append(test_data[i,1])
	else:
		sxt.append(test_data[i,0])
		syt.append(test_data[i,1])
plt.figure(1)
plt.contour(x,y,G,levels=[-9,-8,-7,-6,-5,-4,-3.5,-3,-2])
plt.contour(x,y,H,levels=[-9,-8,-7,-6,-5,-4,-3.5,-3,-2])
plt.contour(x,y,F,levels=[0])
plt.plot(mx,my,'r.')
plt.plot(sx,sy,'b.')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Iso-Probability and Discriminant Function (D) Train Data")
plt.figure(2)
plt.contour(x,y,G,levels=[-8,-7,-6,-5,-4,-3.5,-3,-2])
plt.contour(x,y,H,levels=[-8,-7,-6,-5,-4,-3.5,-3,-2])
plt.contour(x,y,F,levels=[0])
plt.plot(mxt,myt,'g.')
plt.plot(sxt,syt,'y.')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Iso-Probability and Discriminant Function (D) test data")
plt.show()
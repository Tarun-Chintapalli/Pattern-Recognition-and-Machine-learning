import numpy as np
import math
cf=0 ;cs=0 ;i=0 ;ms=0;mf=0;df=[];ds=[];l=[];lf=0;ls=0;lcf=0;lcs=0;fs=0;ff=0
train_label=[]
test_label=[]
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
		df.append(train_data[i])
		cf=cf+1
	else:
		ms=ms+train_data[i]
		ds.append(train_data[i])
		cs=cs+1
sf=np.cov(np.transpose(df))  
ss=np.cov(np.transpose(ds))
mf=mf/cf
ms=ms/cs
prf=float(cf)/float(cf+cs)
prs=float(cs)/float(cs+cf)
oco=(prf*sf)+(prs*ss)
ocov=(np.trace(oco)/2)*np.identity(2)
df=(np.linalg.det(ocov))
ins=np.linalg.inv(ocov)
for i in range (0,90):
	pdff=-math.log(2*math.pi) - (math.log(np.linalg.det(sf)))/2 - (np.matmul(np.transpose(test_data[i]-mf),np.matmul(np.linalg.inv(sf),(test_data[i]-mf))))/2 + math.log(prf)
	pdfs=-math.log(2*math.pi) - (math.log(np.linalg.det(ss)))/2 - (np.matmul(np.transpose(test_data[i]-ms),np.matmul(np.linalg.inv(ss),(test_data[i]-ms))))/2 + math.log(prs)
	if(pdff>=pdfs):
		l.append(1)
	else:
		l.append(0)
for i in range (0,90):
	if(test_label[i]==1):
		if(l[i]==1):
			lf=lf+1
		else:
			fs=fs+1
	if(test_label[i]==0):
		if(l[i]==0):
			ls=ls+1
		else:
			ff=ff+1
for i in range (0,90):
	if(test_label[i]==1):
		lcf=lcf+1
	if(test_label[i]==0):
		lcs=lcs+1
print "\n"
print "When the Covariance is calculated for each class and are different"
print "misclassification rate for '1' is ",float(fs)*100/float(fs+lf),"%"
print "misclassification rate for '0' is ",float(ff)*100/float(ff+ls),"%"
print "Confusion matrix"
print([[lf,fs],[ff,ls]])
print "\n"
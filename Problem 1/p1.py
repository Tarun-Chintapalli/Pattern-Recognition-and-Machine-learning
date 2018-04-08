import numpy as np
import math
cf=0 ;cs=0 ;i=0 ;k=0;ms=0;mf=0;df=[];mx=[];my=[];ds=[];fin=[];lj=[];sx=[];sy=[];l=[];lf=0;ls=0;lcf=0;lcs=0;fs=0;ff=0
train_data = np.genfromtxt('P1_data_train.csv', delimiter=',')
train_label = np.genfromtxt('P1_labels_train.csv', delimiter=',')
test_data = np.genfromtxt('P1_data_test.csv', delimiter=',')
test_label = np.genfromtxt('P1_labels_test.csv', delimiter=',')
for i in range (0, 777):
	if(train_label[i]==5):
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
fcov=(prf*sf)+(prs*ss)
#fcov=fcov*np.identity(64)                                # For covariance with only diagnoal elements
df=(np.linalg.det(fcov))
ins=np.linalg.inv(fcov)
####################################################################################################################### Covariances are of individual classes
for i in range (0,333):
	pdff=-32*math.log(2*math.pi) - (math.log(np.linalg.det(sf)))/2 - (np.matmul(np.transpose(test_data[i]-mf),np.matmul(np.linalg.inv(sf),(test_data[i]-mf))))/2 + math.log(prf)       #  For individual
	pdfs=-32*math.log(2*math.pi) - (math.log(np.linalg.det(ss)))/2 - (np.matmul(np.transpose(test_data[i]-ms),np.matmul(np.linalg.inv(ss),(test_data[i]-ms))))/2 + math.log(prs)       #    Covariance
	if(pdff>=pdfs):
		l.append(5)
	else:
		l.append(6)
for i in range (0,333):
	if(test_label[i]==5):
		if(l[i]==5):
			lf=lf+1
		else:
			fs=fs+1
	if(test_label[i]==6):
		if(l[i]==6):
			ls=ls+1
		else:
			ff=ff+1
for i in range (0,333):
	if(test_label[i]==5):
		lcf=lcf+1
	if(test_label[i]==6):
		lcs=lcs+1
print "\n"
print "When the Covariance is calculated for each class and are different"
print "misclassification rate for '5' is ",float(fs)*100/float(fs+lf),"%"
print "misclassification rate for '6' is ",float(ff)*100/float(ff+ls),"%"
print "Confusion matrix"
print([[lf,fs],[ff,ls]])
print "\n"

##################################################################################################################### Covariances are same
lf=0;ls=0;lcf=0;lcs=0;fs=0;ff=0;l=[]
for i in range (0,333):
	pdff=-32*math.log(2*math.pi) - (math.log(df))/2 - (np.matmul(np.transpose(test_data[i]-mf),np.matmul(ins,(test_data[i]-mf))))/2 + math.log(prf)
	pdfs=-32*math.log(2*math.pi) - (math.log(df))/2 - (np.matmul(np.transpose(test_data[i]-ms),np.matmul(ins,(test_data[i]-ms))))/2 + math.log(prs)
	if(pdff>=pdfs):
		l.append(5)
	else:
		l.append(6)
for i in range (0,333):
	if(test_label[i]==5):
		if(l[i]==5):
			lf=lf+1
		else:
			fs=fs+1
	if(test_label[i]==6):
		if(l[i]==6):
			ls=ls+1
		else:
			ff=ff+1
for i in range (0,333):
	if(test_label[i]==5):
		lcf=lcf+1
	if(test_label[i]==6):
		lcs=lcs+1
print "When the weighted Covariance is calculated for both the classes and is same for each class"
print "misclassification rate for '5' is ",float(fs)*100/float(fs+lf),"%"
print "misclassification rate for '6' is ",float(ff)*100/float(ff+ls),"%"
print "Confusion matrix"
print([[lf,fs],[ff,ls]])
print "\n"

#################################################################################################################### Covariances are same and non diagnoal elements are 0 

lf=0;ls=0;lcf=0;lcs=0;fs=0;ff=0;l=[]
fcov=fcov*np.identity(64) 
df=(np.linalg.det(fcov))
ins=np.linalg.inv(fcov)
for i in range (0,333):
	pdff=-32*math.log(2*math.pi) - (math.log(df))/2 - (np.matmul(np.transpose(test_data[i]-mf),np.matmul(ins,(test_data[i]-mf))))/2 + math.log(prf)
	pdfs=-32*math.log(2*math.pi) - (math.log(df))/2 - (np.matmul(np.transpose(test_data[i]-ms),np.matmul(ins,(test_data[i]-ms))))/2 + math.log(prs)
	if(pdff>=pdfs):
		l.append(5)
	else:
		l.append(6)
for i in range (0,333):
	if(test_label[i]==5):
		if(l[i]==5):
			lf=lf+1
		else:
			fs=fs+1
	if(test_label[i]==6):
		if(l[i]==6):
			ls=ls+1
		else:
			ff=ff+1
for i in range (0,333):
	if(test_label[i]==5):
		lcf=lcf+1
	if(test_label[i]==6):
		lcs=lcs+1
print "When the covariances of the classes are equal (to weighted covariance) and the non diagnoal elements are 0"
print "misclassification rate for '5' is ",float(fs)*100/float(fs+lf),"%"
print "misclassification rate for '6' is ",float(ff)*100/float(ff+ls),"%"
print "Confusion matrix"
print([[lf,fs],[ff,ls]])
print "\n"
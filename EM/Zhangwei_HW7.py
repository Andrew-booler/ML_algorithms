#!/usr/bin/python
import numpy as np
import getopt
import sys
import matplotlib.pyplot as plt
import math

dbpath='./points.dat'
np.random.seed(2)


mode='tied'


#dataloader:
#path: data file path

def dataloader(path,capacity=0.9):
    data=[]
    with open(path,'r') as f:
        data=f.readlines()
    trainset=[]
    testset=[]
    capacity*=len(data)
    for line in data:
        counter=0
        feature=""
        label=""
        if len(line)>0:
            while line[counter]==' ':
                counter+=1
            while line[counter]!=' ':
                feature+=line[counter]
                counter+=1
            while line[counter]==' ':
                counter+=1
            while line[counter]!='\n':
                label+=line[counter]
                counter+=1 
        label=float(label)
        feature=float(feature)   
        if capacity>0:
            capacity-=1
            trainset.append(np.hstack((feature,label)))
        else:
            testset.append(np.hstack((feature,label)))
    return [np.vstack(trainset),np.vstack(testset)]


def train(dataset,w,k):
    

    #M-step

    
    Z=np.sum(w,axis=0)
    cita=Z/dataset.shape[0]
    miu=np.matmul(w.T,dataset)
    cigma=np.zeros((k,2,2))


    for j in range(k):
        miu[j,:]=miu[j,:]/Z[j]
    for j in range(k):
        for i in range(dataset.shape[0]):
            cigma[j]+=w[i][j]*np.matmul(np.reshape((dataset[i]-miu[j]),(2,1)),np.reshape((dataset[i]-miu[j]),(1,2)))
        cigma[j]=cigma[j]/Z[j]
    if mode=='tied':

        for i in range(k):
            cigma[i]=cita[i]*cigma[i]

        cigma_mean = np.sum(cigma, axis=0,keepdims=True)
        for i in range(k):
            cigma[i] = cigma_mean    


            

            
    #E-step
    EZ=np.zeros(dataset.shape[0])
    likely=0
    counter=0
    for i in range(dataset.shape[0]):
    
        for j in range(k):
            temp=np.matmul(np.matmul((dataset[i]-miu[j]),np.linalg.inv(cigma[j])),(dataset[i]-miu[j]).T)

            w[i][j]=1/(2*math.pi)/math.sqrt(np.linalg.det(cigma[j]))*np.e**(-0.5*temp)*cita[j]        
            EZ[counter]+=w[i][j]
        for j in range(k):
            w[i][j]/=EZ[counter]
        
        counter+=1
    

        
        
    
            
    #print "training stopped"
    return [np.sum(np.log(EZ))/dataset.shape[0],w,cita,miu,cigma]
def test(dataset,w,cita,miu,cigma,k):
    EZ=np.zeros(dataset.shape[0])
    likely=0
    counter=0
    for i in range(dataset.shape[0]):
    
        for j in range(k):
            temp=np.matmul(np.matmul((dataset[i]-miu[j]).T,np.linalg.inv(cigma[j])),(dataset[i]-miu[j]))
            w[i][j]=1/math.sqrt((2*math.pi)**k*np.linalg.det(cigma[j]))*np.e**(-0.5*temp)*cita[j]        
            EZ[counter]+=w[i][j]
        for j in range(k):
            w[i][j]/=EZ[counter]
        counter+=1
    return np.sum(np.log(EZ))/dataset.shape[0], w

def main(argv):
    #print "training......"
    epochs=1
    capacity=1.0
    k= input("Please enter the mixtures number: ")
    k= int(k)
    
    trainset,testset = dataloader(dbpath)
    trw=np.random.random((trainset.shape[0],k))
    tew=np.random.random((testset.shape[0],k))
    #lamb = np.ones(k)/k
    #u = np.zeros((k,2))
    #cov = np.random.rand(k,2,2)
    #for x in xrange(k):
        #u[x] = np.mean(trainset,axis=0)+0.01*np.random.rand()
        #cov[x] = np.identity(2)
    #for l in lamb:
        #l=1/k
    

    cap_list=np.arange(1,101)    
    trlikelihood=np.zeros(100)
    telikelihood=np.zeros(100)
    while epochs<=100:
               
        trlikelihood[epochs-1],trw,a,b,c = train(trainset, trw,k)
        telikelihood[epochs-1],tew= test(testset,tew,a,b,c,k)
        epochs+=1 
        
        #print lamb
        #print u
        
    plt.plot(cap_list,trlikelihood)
    plt.plot(cap_list,telikelihood,color="red")
    plt.ylabel('likelihood')
    plt.xlabel('iteration')
    plt.show()        




if __name__ =="__main__":
    main(sys.argv)


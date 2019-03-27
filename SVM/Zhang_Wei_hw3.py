#!/usr/bin/python
import numpy as np
import getopt
import sys
import matplotlib.pyplot as plt

dbpath='./adult'
train_file='a7a.train'
test_file='a7a.test'
dev_file='a7a.dev'

max_epoch=10



def sign(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    return 0
#dataloader:
#path: data file path
#return: a n*124 array with first column as the label the rest as features
def dataloader(path,capacity=1):
    data=[]
    with open(path,'r') as f:
        data=f.readlines()
    dataset=[]
    capacity*=len(data)
    for line in data:
        if capacity>0:
            capacity-=1
        segments= line.split(" ")[:-1]
        label=np.array([int(segments[0])])
        features=np.zeros(124)
        features[-1]=1.0
        for seg in segments[1:]:
            [index, value] = seg.split(":")
            features[int(index)-1]=float(value)
        dataset.append(np.hstack((label, features)))
    return np.vstack(dataset)

def train(dataset,weight,iter,C):
    lr=0.1
    for epoch in range(iter):
        for tran in dataset:
            pred=np.dot(weight, tran[1:])*tran[0]
            if 1-pred > 0:
                weight[:-1] -= 1.0*weight[:-1]*lr/len(dataset)-lr*C*tran[0]*tran[1:-1]
                weight[-1] += lr*C*tran[0]
            else:
                weight[:-1]  -= lr*weight[:-1]/len(dataset)
    #print "training stopped"
    return weight

def test(dataset,weight):
    match = 0.0
    total = dataset.shape[0]
    for tran in dataset:
        pred=np.dot(weight, tran[1:])
        if sign(pred) == tran[0]:
            match+=1
    return match/total

def main(argv):
    #print "training......"
    epochs=1
    capacity=1.0
    if len(argv)>1:
        opts, args= getopt.getopt(argv[1:], '',['epochs=','capacity='])
        for opt,value in opts:
            if opt=='--epochs':
                epochs=int(value)
            else:
                capacity=float(value)
        weight = np.zeros(124)
        trainset = dataloader(dbpath + '/' + train_file)
        testset = dataloader(dbpath + '/' + test_file)
        devset = dataloader(dbpath + '/' + dev_file)
        
        print "EPOCHS: ", epochs
        print "CAPACITY: ", capacity
        weight = train(trainset, weight,epochs,capacity)
        print "TRAINING_ACCURACY: ", test(trainset, weight)
        print "TEST_ACCURACY: ", test(testset, weight)
        print "DEV_ACCURACY: ", test(devset, weight)
        
        out="["+str(weight[-1])
        for w in weight[:-1]:
            out+=", "
            out+=str(w)
            
        print "FINAL_SVM: ",out,"]"
    else:
        trainset = dataloader(dbpath + '/' + train_file)
        testset = dataloader(dbpath + '/' + test_file)
        valset= dataloader(dbpath + '/' + dev_file)
        epochs=5
        test_accuracy=[]
        val_accuracy=[]
        cap_list=[]
        for pcapacity in range(21):
            capacity=10**(float(pcapacity)/3-3)
            cap_list.append(float(pcapacity)/3-3)
            weight = np.zeros(124)
            weight = train(trainset, weight,epochs,capacity)
            test_accuracy.append(test(testset,weight))
            val_accuracy.append(test(valset,weight))
        plt.plot(cap_list,test_accuracy)
        plt.plot(cap_list,val_accuracy,color="red")
        plt.ylabel('accuracy')
        plt.xlabel('capacity')
        plt.show()

if __name__ =="__main__":
    main(sys.argv)


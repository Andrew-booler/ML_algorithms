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
def dataloader(path):
    data=[]
    with open(path,'r') as f:
        data=f.readlines()
    dataset=[]
    for line in data:
        segments= line.split(" ")[:-1]
        label=np.array([int(segments[0])])
        features=np.zeros(124)
        features[-1]=1.0
        for seg in segments[1:]:
            [index, value] = seg.split(":")
            features[int(index)-1]=float(value)
        dataset.append(np.hstack((label, features)))
    return np.vstack(dataset)

def train(dataset,weight,iter):
    lr=1
    for epoch in range(iter):
        for tran in dataset:
            pred=np.dot(weight, tran[1:])
            if sign(pred) != tran[0]:
                weight -= lr*tran[0]*tran[1:]
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
    if len(argv)>1:
        opts, args= getopt.getopt(argv[1:], '',['epochs=','capacity='])
        for opt,value in opts:
            if opt=='--epochs':
                weight = np.zeros(124)
                trainset = dataloader(dbpath + '/' + train_file)
                weight = train(trainset, weight,int(value))
                testset = dataloader(dbpath + '/' + test_file)
                print "Test accuracy:", test(testset, weight)
                print "Feature weights (bias last): ", weight
    else:
        trainset = dataloader(dbpath + '/' + train_file)
        accuracy=[]
        weight = np.zeros(124)
        for iter in range(20):
            trainset = dataloader(dbpath + '/' + train_file)
            weight = train(trainset, weight,1)
            valset= dataloader(dbpath + '/' + dev_file)
            accuracy.append(test(valset,weight))
        plt.plot(accuracy)
        plt.ylabel('accuracy')
        plt.xlabel('iterations')
        plt.show()

if __name__ =="__main__":
    main(sys.argv)


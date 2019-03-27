#!/usr/bin/python
import numpy as np
import math
import matplotlib.pyplot as plt
from math import pi,sqrt,exp,pow
from numpy.linalg import det, inv



datapath="./points.dat"
dim = 2
iteration = 30

def normalize(X):
    res=[]
    for item in X:
        res.append(item/np.sum(item))
    return np.array(res)

def gauss2D(x, mean, cov):
    z = -np.dot(np.dot((x-mean).T,inv(cov)),(x-mean))/2.0
    temp = pow(sqrt(2.0*pi),len(x))*sqrt(det(cov))
    return (1.0/temp)*exp(z)

class GaussianHMM():

    def __init__(self, n_state=1, x_size=1):
        self.n_state = n_state
        self.x_size = x_size
        self.start_prob = np.ones(n_state) * (1.0 / n_state)  
        self.transmat_prob = np.ones((n_state, n_state)) * (1.0 / n_state)  
        self.cov_record=0
        self.emit_means = np.zeros((n_state, x_size))      
        self.emit_covars = np.zeros((n_state, x_size, x_size)) 
        for i in range(n_state): self.emit_covars[i] = np.eye(x_size)  

    def _init(self,X):

        self.emit_means = 0.01*np.random.rand(self.n_state, self.x_size)
	self.emit_means /= np.sum(self.emit_means)
        for i in range(self.n_state):
            self.emit_covars[i] = np.cov(X.T) + 0.01 * np.eye(len(X[0]))

    def emit_prob(self, x): 
        prob = np.zeros((self.n_state))
        for i in range(self.n_state):
            if abs(self.cov_record-self.emit_covars[i][0][0])>0.05:

                self.cov_record = self.emit_covars[i][0][0]
            prob[i]=gauss2D(x,self.emit_means[i],self.emit_covars[i])
        return prob



    def emit_prob_updated(self, X, post_state): 
        for k in range(self.n_state):
            for j in range(self.x_size):

                self.emit_means[k][j] = np.sum(post_state[:,k] *X[:,j]) / np.sum(post_state[:,k])

            X_cov = np.dot((X-self.emit_means[k]).T, (post_state[:,k]*(X-self.emit_means[k]).T).T)
            self.emit_covars[k] = X_cov / np.sum(post_state[:,k])
            if det(self.emit_covars[k]) == 0: 

                self.emit_covars[k] = self.emit_covars[k] + 0.01*np.eye(len(X[0]))

    def train(self, X, Z_seq=np.array([])):


        X_length = len(X)
        Z=Z_seq
        alpha = 0
        post_state = 0
        # E step
        alpha, c = self.forward(X, Z)  # P(x,z)
        beta = self.backward(X, Z, c)  # P(x|z)

        post_state = alpha * beta
        post_adj_state = np.zeros((self.n_state, self.n_state))  
        for i in range(X_length):
            if i == 0: continue
            if c[i]==0: continue
            post_adj_state += (1 / c[i])*np.outer(alpha[i - 1],beta[i]*self.emit_prob(X[i]))*self.transmat_prob

            # M step
        self.start_prob = post_state[0] / np.sum(post_state[0])
        for k in range(self.n_state):
            self.transmat_prob[k] = post_adj_state[k] / np.sum(post_adj_state[k])
        self.emit_prob_updated(X, post_state)
        return post_state, post_adj_state


    def forward(self, X, Z):
        X_length = len(X)
        c = np.zeros(X_length)
        alpha = np.zeros((X_length, self.n_state)) 
        alpha[0] = self.emit_prob(X[0]) * self.start_prob * Z[0] 
        c[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / c[0]
        for i in range(X_length):
            if i == 0: continue
            alpha[i] = self.emit_prob(X[i]) * np.dot(alpha[i - 1], self.transmat_prob) * Z[i]
            c[i] = np.sum(alpha[i])
            if c[i]==0: continue
            alpha[i] = alpha[i] / c[i]
        return alpha, c

    def backward(self, X, Z, c):
        X_length = len(X)
        beta = np.zeros((X_length, self.n_state))
        beta[X_length - 1] = np.ones((self.n_state))

        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            beta[i] = np.dot(beta[i + 1] * self.emit_prob(X[i + 1]), self.transmat_prob.T) * Z[i]
            if c[i+1]==0: continue
            beta[i] = beta[i] / c[i + 1]
        return beta

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
    trainset = np.array(trainset)
    testset = np.array(testset)
    return [np.vstack(trainset),np.vstack(testset)]

hidden_state=input("Please enter the number of hidden states: ")
model = GaussianHMM(hidden_state, dim)
trainset,testset = dataloader(datapath)
model._init(trainset)



tr_log_like=[]
te_log_like=[]
for iter in range(iteration):
    print "iteration: "+str(iter+1)
    tr_log_like.append(0)
    te_log_like.append(0)
#print alpha
    Zi = np.ones((len(trainset), hidden_state))
    r, si=model.train(trainset,Zi)
    tr_log_like[iter]=np.dot(r[0],np.log(normalize(r)[0])+np.sum(si*model.transmat_prob))
    te_log_like[iter]=tr_log_like[iter]
    for i in range(900):
        tr_log_like[iter]+=np.dot(r[i],np.log(model.emit_prob(trainset[i])))
        if i < 100:
            te_log_like[iter]+=np.dot(r[i],np.log(model.emit_prob(testset[i])))
    tr_log_like[iter]/=1000
    te_log_like[iter]/=1000
title =  'train state number:'+str(hidden_state)
plt.figure(1)
plt.subplot(211)
plt.plot(tr_log_like)
plt.title(title)
title =  'dev state number:'+str(hidden_state)
plt.subplot(212)
plt.plot(te_log_like)
plt.title(title)
plt.show()



#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import random
import itertools
from sklearn import preprocessing as pp
import copy


class EPSDP():
    def __init__(self, Problem, Penaliser, alpha, **kwargs): #param: rank, precision, maxsteps, maxrounds, rounding
        
        self.Problem = Problem
        self.Cost_Matrix = Problem.Matrix() #i.e. we maximize   <x, Cost_Matrix x>
        self.Type = Penaliser
        self.alpha = alpha
        
        if 'rank' in kwargs.keys():
            self.rank = kwargs['rank']
        else:
            self.rank = int(np.sqrt(2*len(self.Cost_Matrix))) + 1 #if not specified, set to BVB value
            
        if 'rounding' in kwargs.keys(): #may be: GW, Random, D-Wave, Greedy
            self.rounding = kwargs['rounding']
        else:
            self.rounding = "GW"
            
        if 'sigma' in kwargs.keys(): #may be: GW, Random, D-Wave, Greedy
            self.sigma = kwargs['sigma']
        else:
            self.sigma = 10
            
        if 'gamma' in kwargs.keys(): #may be: GW, Random, D-Wave, Greedy
            self.gamma = kwargs['gamma']
        else:
            self.gamma = 1.5    
        
        if 'maxsteps' in kwargs.keys():
            self.maxsteps = kwargs['maxsteps']
        else:
            self.maxsteps = 500
            
        if 'maxrounds' in kwargs.keys():
            self.maxrounds = kwargs['maxrounds']
        else:
            self.maxrounds = 100
        
        if 'precision' in kwargs.keys():
            self.precision = kwargs['precision']
        else:
            self.precision = 0.01
            
    def __str__(self):
        self.Solve()
            
        self.GetUB()
        self.GetLB()
        return "SDP lower_bound = %s, SDP upper_bound = %s"% (self.lower_bound, self.upper_bound)
    
    def GetUB(self):        
        self.upper_bound = np.trace(np.matmul(self.Cost_Matrix, np.matmul(np.transpose(self.V), self.V)))
        
    def GetLB(self):
        if (self.rounding == "GW"):
            self.assignment = GWRounding(self.Cost_Matrix, self.V, maxrounds=self.maxrounds)
            self.assignment = cut_last(self.assignment) #make extra variable equal to 1
            self.lower_bound = (self.Problem).Evaluate(self.assignment) #bqp.Evaluate(self.Problem, self.assignment)
        elif (self.rounding == "Random"):
            self.assignment = RandomCut(len(self.Cost_Matrix), maxrounds=self.maxrounds)
            self.assignment = cut_last(self.assignment)#make extra variable equal to 1
            self.lower_bound = (self.Problem).Evaluate(self.assignment)#bqp.Evaluate(self.Problem, self.assignment)
        else:    
            print("Method is not supported")
            
            
    def Solve(self):
        #initialize randomly on sphere
        C = self.Cost_Matrix
        if (np.count_nonzero(C) == 0):
            self.V = np.zeros((self.rank, len(C)))
            return None
        initV = np.random.normal(0, 1, (self.rank, len(C)))
        
        #determine columns that are nonzero
        nonzero = list(np.where(self.Cost_Matrix.any(axis=1))[0]) #(np.where(self.Cost_Matrix != 0))[1]
        #restrict  C and V to non-zero columns
        C = self.Cost_Matrix[np.ix_(nonzero, nonzero)]
        nnzV = initV[np.ix_(list(range(self.rank)), nonzero)]

        #normalize columns
        nnzV = np.transpose(pp.normalize(np.transpose(nnzV), norm='l2'))
        #self.NormalizeColumns()

        #specify step-size
        step = 1/np.linalg.norm(C) #Lipschitz
        
        la = self.sigma
        ga = self.gamma
        
        for outer_steps in range(10):
            for steps in range(self.maxsteps):
                gradient = 2*np.matmul(nnzV, C) - la*self.PenGrad(nnzV); 
                nnzV = nnzV + step*(gradient)
                #self.V += np.random.normal(0, 1, (self.rank, len(C)))/1000   # do i need this?
                nnzV = np.transpose(pp.normalize(np.transpose(nnzV), norm='l2'))
                #self.NormalizeColumns(
            
            la = la*ga
            
         #return to full-size by filling with zeros or randomly
        self.V = np.zeros((self.rank, len(self.Cost_Matrix)))
        self.V[np.ix_(list(range(self.rank)), nonzero)] = nnzV   
    
    def PenGrad(self, V):
        alpha = self.alpha
        
        U1, d, U2 = np.linalg.svd(V, full_matrices=False)
        D = np.diag(d)
         
        #X = V.T*V
        
        if (self.Type == "Tsallis"):
            
            return (alpha/(1-alpha))*(np.dot(U1, np.dot(np.linalg.matrix_power(D, 2*alpha-1),U2))/np.power(np.trace(np.dot(D, D)), alpha) -  np.trace(np.linalg.matrix_power(D, 2*alpha))*V/np.power(np.trace(np.dot(D, D)), alpha+1) )
        
        elif (self.Type == "Renyi"):
            return 0*V
            
        elif (self.Type == "Neumann"):
            return 0*V
        else:
            return 0*V
#--------------------------------------------------------------------------------------------        
        
def cut_last(assignment_plus):
    n = len(assignment_plus[0])-1
    
    assignment = copy.deepcopy(assignment_plus)
    assignment = assignment*assignment[0][len(assignment[0])-1]
    assignment = list(assignment[0])
    assignment.pop()
    assignment = np.array(assignment)
    assignment = assignment.reshape((1, n))
    
    return assignment
    
    
def GWRounding(A, V, **kwargs):
    
    
    if 'maxrounds' in kwargs.keys():
        maxrounds = kwargs['maxrounds']
    else:
        maxrounds = 100
        
    
    n = len(V[0, :])
    k = len(V[:, 0])
    
    cutVal = 0.0;
    cutAssignment = np.zeros((1, n));

    for cut_trials in range(maxrounds):
        
        r = np.random.normal(0, 1, (1, k))
        r = r[0]/np.sqrt(np.dot(r[0], r[0]));
        
        cut = np.zeros((1, n));
        for cut_iter in range(n):
            cut[0][cut_iter] =  np.sign(np.dot(r, V[:, cut_iter]))
        
        cutValnew = np.dot(cut, np.matmul(A, cut.T))

        if (cut_trials == 1):
            cutVal = cutValnew
            cutAssignment = cut    
        elif (cutValnew > cutVal):
            cutVal = cutValnew
            cutAssignment = cut
    
    for i in range(n):
        if (cutAssignment[0][i] == 0):
            cutAssignment[0][i] = np.sign(np.random.normal(0, 1))    
        
    
    #return a row-vector
    cutAssignment = cutAssignment.astype(int)
    
    return cutAssignment    



def RandomCut(A, **kwargs):
    
    if 'maxrounds' in kwargs.keys():
        maxrounds = kwargs['maxrounds']
    else:
        maxrounds = 100
    
    bestval = 0.0
    bestcut = np.ones((1, len(A)))
    
    for i in range(maxrounds):
        v = np.random.normal(0, 1, (1, len(A)))[0]
        cut = [np.sign(i) for i in v] 
        cut = np.reshape(cut, (1, len(cut)))
        cutval = np.trace(np.matmul(A, np.dot(cut.T, cut)))

        if (cutval > bestval):
            bestcut = cut
            bestval = cutval
    
    return (bestval, bestcut)    
        


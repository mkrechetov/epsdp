import numpy as np
import random
import itertools
from sklearn import preprocessing as pp
import copy

from bqp_class import *
from solution_class import *

class SDP(): #use methods to solve
    def __init__(self, Problem, **kwargs): #param: rank, precision, maxsteps, maxrounds, rounding
        self.Problem = Problem
        self.Cost_Matrix = Problem.Matrix() #i.e. we maximize   <x, Cost_Matrix x>
        
        if 'rank' in kwargs.keys():
            self.rank = kwargs['rank']
        else:
            self.rank = int(np.sqrt(2*len(self.Cost_Matrix))) + 1 #if not specified, set to BVB value
            
        if 'rounding' in kwargs.keys(): #may be: GW, Random, D-Wave, Greedy
            self.rounding = kwargs['rounding']
        else:
            self.rounding = "GW"   
        
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
            
        if 'method' in kwargs.keys():
            self.method = kwargs['method']
        else:
            self.method = "B-M"
            
    def __str__(self):
        if (self.method == "B-M"):
            self.SolveBM()
        else:
            self.SolveSDP()
            
        self.GetUB()
        self.GetLB()
        return "SDP lower_bound = %s, SDP upper_bound = %s, steps = %s"% (self.lower_bound, self.upper_bound, self.Allsteps)
    
    def Solve(self):
        if (self.method == "B-M"):
            self.SolveBM()
        else:
            self.SolveSDP()
            
        self.GetUB()
        self.GetLB()
        return Solution(self.lower_bound, self.upper_bound, self.assignment, self.assignment, self.V)
    
    def SolveBM(self):
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
    
        for steps in range(self.maxsteps):
            gradient = 2*np.matmul(nnzV, C); 
            nnzV = nnzV + step*gradient;
            #self.V += np.random.normal(0, 1, (self.rank, len(C)))/1000   # do i need this?
            nnzV = np.transpose(pp.normalize(np.transpose(nnzV), norm='l2'))
            #self.NormalizeColumns()
            
            #update stopping variables
            dualvar = np.zeros((1, len(C)))
            M = np.matmul(nnzV, C)
            dualvar = M[0, :]/nnzV[0, :]
            
            #fix incorrect dualvars --   WEAK point
            #print(np.argwhere(np.isnan(dualvar)))
            dual_undef = np.argwhere(np.isnan(dualvar))
            dual_undef = np.reshape(dual_undef, (1, len(dual_undef)))[0]
            dual_undef = dual_undef.tolist() #list of indices to be determined
            dualvar[np.ix_(dual_undef)] = np.zeros((len(dual_undef,)))
            
            #optimality conditions, see Absil et al
            condition1 = (np.linalg.norm(M-nnzV*np.transpose(dualvar), ord=np.inf)<self.precision)
            condition2 = (max(np.linalg.eigvals(C-np.diag(dualvar)))>-self.precision)
            if (condition1 & condition2):
                #print("BMSDP converged!")
                self.Allsteps = steps
                break
            self.Allsteps = steps    
         
        #return to full-size by filling with zeros or randomly
        self.V = np.zeros((self.rank, len(self.Cost_Matrix)))
        self.V[np.ix_(list(range(self.rank)), nonzero)] = nnzV
                
    def ChangeObjective(self, NewObjective):
        newSDP = copy.deepcopy(self)
        newSDP.Problem = NewObjective
        newSDP.Cost_Matrix = NewObjective.Matrix()
        newSDP.upper_bound = None
        newSDP.assignment = None
        newSDP.lower_bound = None
        return newSDP
   
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
            
    def SolveSDP(self):
        import cvxpy as cvx
        #solve cvxpy sdp
        n = len(self.Cost_Matrix)
        X = cvx.Variable((n, n), symmetric= True) #PSD=True 
        obj = cvx.Maximize(cvx.trace(self.Cost_Matrix*X))
        cons = [cvx.diag(X) == np.ones((n))]
        cons += [X >> np.zeros((n,n))]
        prob = cvx.Problem(obj, cons)
        prob.solve(solver=cvx.SCS, eps=1e-5)
        
        self.V = np.linalg.cholesky(X.value + 0.001*np.diag(np.ones(n))).T
        self.Allsteps = -1
        
    
    def NormalizeColumns(self):
        for col in range(len(self.Cost_Matrix)):
            column = self.V[np.ix_(range(self.rank), [col])]
            if (abs(sum(column)) >= 0.01): #np.zeros((self.rank,))
                self.V[np.ix_(range(self.rank), [col])] = pp.normalize(column, norm='l2')
            
    
#--------------------------------------------------------------------------------------------------------------------------    
     
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

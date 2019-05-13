import numpy as np
from scipy.sparse import csr_matrix 
import networkx as nx
import json
import copy

import sys
from gurobipy import *

class bqp: #to read and to rewrite problems, return matrix
      
    def __init__(self, init_type, **kwargs): #init_type: {Directly, Random, From File, Random Tree, Random Chimera}
        
        
        if (init_type == "Directly"):
            if 'Const' in kwargs.keys():
                Const = kwargs['Const']
            else:
                Const = 0
            
            if 'Pot' in kwargs.keys():
                Pot = kwargs['Pot']
            else:
                Pot = np.zeros((len(kwargs['Inter'])))
                
            (self.Inter, self.Pot, self.Const) = (kwargs['Inter'], Pot, Const)
        elif (init_type == "Random"):
            
            if 'n' in kwargs.keys():
                n = kwargs['n']
            else:
                print("Should specify number of vertices n=")
                
            if 'p' in kwargs.keys():
                p = kwargs['p']
            else:
                print("Should specify the probability of an edge p=")    
            
            if 'seed' in kwargs.keys():
                seed = kwargs['seed']
            else:
                seed = None

            self.G = nx.gnp_random_graph(n, p, seed)
            A = csr_matrix.todense(nx.adjacency_matrix(self.G))

            (self.Inter, self.Pot, self.Const) = (A, np.zeros((n)), 0)#Laplacian(A)/4
        
        elif (init_type == "RandomMaxCut"):
            
            if 'n' in kwargs.keys():
                n = kwargs['n']
            else:
                print("Should specify number of vertices n=")
                
            if 'p' in kwargs.keys():
                p = kwargs['p']
            else:
                print("Should specify the probability of an edge p=")    
            
            if 'seed' in kwargs.keys():
                seed = kwargs['seed']
            else:
                seed = None

            self.G = nx.gnp_random_graph(n, p, seed)
            L = csr_matrix.todense(nx.laplacian_matrix(self.G))
            
            
            (self.Inter, self.Pot, self.Const) = (L/4, np.zeros((n)), 0)#Laplacian(A)/4
            
        elif (init_type == "From File"):#----------------add self.G!!!--------------
            
            if 'filename' in kwargs.keys():
                filename = kwargs['filename']
            else:
                print("You should specify the filename!")
                
            import os
            name, extension = os.path.splitext(filename)

            if (extension == '.json'):
                (self.Inter, self.Pot, self.Const) = BQPJSON(filename)
                self.Inter = 1*self.Inter
                self.Pot = 1*self.Pot
                self.Const = 1*self.Const
            elif (extension == '.mat'):
                #retrieve a dense graph from .mat file

                with open(filename) as f:
                    data = f.readlines()

                n = int(data[0])

                Interactions = np.zeros((n, n))

                for rows in range(1, n+1):
                    row = data[rows].split()
                    for col in range(n):
                        Interactions[rows-1, col] = int(row[col])
                
                problem = bqp(init_type = "Directly", Inter = Interactions, Pot = np.zeros((n)))
                problem = problem.SpinFromBool()
                
                (self.Inter, self.Pot, self.Const) = (problem.Inter, problem.Pot, problem.Const)
                
            elif (extension == '.sparse'):
                #retrieve a sparse graph from .sparse file

                with open(filename) as f:
                    data = f.readlines()
                
                inf = data[0].split()
                n = int(inf[0])
                m = int(inf[1])

                Interactions = np.zeros((n, n))

                for nnz in range(m):
                    row = data[nnz+1].split()
                    Interactions[int(row[0])-1, int(row[1])-1] = int(row[2])
                    Interactions[int(row[1])-1, int(row[0])-1] = int(row[2])
                
                problem = bqp(init_type = "Directly", Inter = Interactions, Pot = np.zeros((n)))
                problem = problem.SpinFromBool()
                
                (self.Inter, self.Pot, self.Const) = (problem.Inter, problem.Pot, problem.Const)
            else:
                print("Wrong File Extension")
                
        elif (init_type == "Chimera"):
            import dwave_networkx as dnx

            self.G = dnx.chimera_graph(kwargs['M'], kwargs['N'], kwargs['L'])
            A = csr_matrix.todense(nx.adjacency_matrix(self.G))   
           
            (self.Inter, self.Pot, self.Const) = (A, np.zeros((len(A))), 0)        #Laplacian(A)/4
                
        elif (init_type == "Random Chimera"):
            import dwave_networkx as dnx

            G = dnx.chimera_graph(kwargs['M'], kwargs['N'], kwargs['L'])
            A = csr_matrix.todense(nx.adjacency_matrix(G))
                
            if 'p' in kwargs.keys():
                p = kwargs['p']
            else:
                print("Should specify the probability of an edge p=")    
            
            if 'seed' in kwargs.keys():
                seed = kwargs['seed']
            else:
                seed = None
            
            n = 2*kwargs['M']*kwargs['N']*kwargs['L']
            
            H = nx.gnp_random_graph(n, kwargs['p'], seed)
            B = csr_matrix.todense(nx.adjacency_matrix(H))
            
            A = np.multiply(A,B)
            self.G = nx.from_numpy_matrix(A)
            
            (self.Inter, self.Pot, self.Const) = (Laplacian(A)/4, np.zeros((len(A))), 0)
            
        elif (init_type == "RAN"):
            
            import dwave_networkx as dnx

            G = dnx.chimera_graph(kwargs['M'], kwargs['N'], kwargs['L'])
            A = csr_matrix.todense(nx.adjacency_matrix(G))
                
            if 'p' in kwargs.keys():
                p = kwargs['p']
            else:
                print("Should specify the probability of an edge p=")    
            
            if 'seed' in kwargs.keys():
                seed = kwargs['seed']
            else:
                seed = None
            
            n = 2*kwargs['M']*kwargs['N']*kwargs['L']
            
            B = np.random.choice([-1, 1], (n, n), p=[p, 1-p])
            #print("B equals to:", B)
            A = np.multiply(A,B)
            self.G = nx.from_numpy_matrix(A)
            
            (self.Inter, self.Pot, self.Const) = (A, np.zeros((len(A))), 0)#Laplacian(A)/4
 
        
        elif (init_type == "Random Tree"):
            
            if 'seed' in kwargs.keys():
                seed = kwargs['seed']
            else:
                seed = None
    
            if 'n' in kwargs.keys():
                n = kwargs['n']
            else:
                n = random.randint(10, 100)

            self.G = nx.random_tree(n, seed)
            A = csr_matrix.todense(nx.adjacency_matrix(self.G))
            
            (self.Inter, self.Pot, self.Const) = (Laplacian(A)/4, np.zeros((n)), 0)
            
        
    def __str__(self):
        return "Matrix of Interactions =\n %s,\n Vector of Potentials =\n %s,\n Constant = %s"% (self.Inter, self.Pot, self.Const)    
    
    def __mul__(self, other): #hadamard product of the data
        
        resInter = np.multiply(self.Inter,other.Inter)
        resPot = np.multiply(self.Pot,other.Pot)
        resConst = np.multiply(self.Const,other.Const)
        
        return bqp(init_type = "Directly", Inter = resInter, Pot = resPot, Const = resConst)
    
    def Evaluate(self, assignment):
        return np.trace(np.matmul(self.Inter, np.dot(assignment.T, assignment))) + np.dot(self.Pot, assignment.T) + self.Const
            
        
    #converts to quadratic problem without linear and constant terms
    #s.t. we will have  maximization  <x, C x> subject to spin constraints    
    def Matrix(self):
        
        n = len(self.Inter)
        
        C = np.zeros((n + 1, n + 1))
        C[np.ix_(list(range(n)),list(range(n)))] = self.Inter
        half_Pot = [i/2 for i in self.Pot]
        C[np.ix_([n],list(range(n)))] = half_Pot
        C[np.ix_(list(range(n))), [n]] = half_Pot

        C[np.ix_([n], [n])] = self.Const
            
        return C
    
    def Merge(self, partial_assignment, curr_known):
        
        variables = copy.deepcopy(curr_known)
        variables.reshape(1, len(variables))
        
        ind_unknown = np.argwhere(np.isnan(variables))
        ind_unknown = np.reshape(ind_unknown, (1, len(ind_unknown)))[0]
        ind_unknown = ind_unknown.tolist() 
        
        variables[np.ix_(ind_unknown)] = partial_assignment.reshape(1, (len(partial_assignment[0])))
        return variables.reshape((1, len(curr_known)))
    
    def InducedProblem(self, variables): #variables is (1, n) array of +-1s and nans.
        
        ind_known = np.argwhere(~np.isnan(variables))
        ind_known = np.reshape(ind_known, (1, len(ind_known)))[0]
        ind_known = ind_known.tolist()    

        ind_unknown = np.argwhere(np.isnan(variables))
        ind_unknown = np.reshape(ind_unknown, (1, len(ind_unknown)))[0]
        ind_unknown = ind_unknown.tolist() 

        
        #induced Interaction   
        induced_Inter = self.Inter[np.ix_(ind_unknown,ind_unknown)]    
        known_Inter = self.Inter[np.ix_(ind_known,ind_known)]
        cross_Inter = self.Inter[np.ix_(ind_known, ind_unknown)]
        
        var_defined = variables[np.argwhere(~np.isnan(variables))]
        var_defined = np.reshape(var_defined, (1, len(var_defined)))
        
        known_Potentials = self.Pot[np.argwhere(~np.isnan(variables))]
        known_Potentials = np.reshape(known_Potentials, (1, len(known_Potentials)))
                
        unknown_Potentials = self.Pot[np.argwhere(np.isnan(variables))]
        unknown_Potentials = np.reshape(unknown_Potentials, (1, len(unknown_Potentials)))
        
        linear_term = 2*np.matmul(var_defined,cross_Inter) + unknown_Potentials
        
        lyft = np.trace(np.matmul(known_Inter, np.dot(var_defined.T, var_defined))) + np.dot(var_defined, known_Potentials.T) + self.Const
        
        return bqp(init_type = "Directly", Inter = induced_Inter, Pot = linear_term, Const = lyft)
    
    
    def BoolFromSpin(self):
        
        n = len(self.Inter)
        
        bool_Inter = 4*self.Inter
        
        bool_Pot = -4*np.matmul(np.ones(n), self.Inter) + 2*self.Pot
        b = np.array(bool_Pot)
        bool_Pot = b.flatten()
        
        bool_Const = np.trace(np.matmul(self.Inter, np.dot(np.ones((1, n)).T, np.ones((1, n))))) + self.Const - sum(self.Pot)
        
        result = bqp(init_type = "Directly", Inter = bool_Inter, Pot = bool_Pot, Const = bool_Const)
        #result.
        ()
        
        return result
    
    def SpinFromBool(self):
        
        n = len(self.Inter)
        
        spin_Inter = self.Inter/4
        
        spin_Pot = self.Pot/2 + np.matmul(np.ones(n), self.Inter/2)
        b = np.array(spin_Pot)
        spin_Pot = b.flatten()
        
        spin_Const = np.dot(np.ones((1,n)), self.Pot.T/2) + np.trace(np.matmul(self.Inter/4, np.dot(np.ones((1, n)).T, np.ones((1, n)))))
        
        
        
        result = bqp(init_type = "Directly", Inter = spin_Inter, Pot = spin_Pot, Const = spin_Const[0])
        #result.Format()
        
        return result            
                   
        
        
    def Solve(self, **kwargs):
        
        n = len(self.Inter)
        
        m = Model("bqp")
        #m.setParam('OutputFlag', False)
        
        if 'baseline' in kwargs.keys():
            m.setParam('BestObjStop', kwargs['baseline'])
            m.setParam('BestBdStop', kwargs['baseline'])
            
            
        
        problem = self.BoolFromSpin()
        #print(problem)
        
        A = problem.Inter
        b = problem.Pot
        c = problem.Const
        #print(b)
        n = len(A)

        vars = []
        for j in range(n):
            vars.append(m.addVar(vtype=GRB.BINARY))
            #m.addConstr(vars[j]*vars[j] == 1)

        # Populate objective
        obj = QuadExpr()
        for i in range(n):
            for j in range(n):
                if A[i, j] != 0:
                    obj += A[i, j]*vars[i]*vars[j]
        for j in range(n):
            if b[j] != 0:
                obj += b[j]*vars[j]
        obj += c

        m.setObjective(obj, GRB.MAXIMIZE)

        # Solve
        m.optimize()
        
        return m
    
    
    def SolveLP(self):# Fix!!!!
        
        n = len(self.Inter)
        
        #mask = bqp("Random", n=n, p=densities[prob_iter])
        
        #problem = problem*mask
        
        #----solve by gurobi
        

        # Create a new model
        m = Model("bqp")
        m.setParam('OutputFlag', False)
        problem = self.BoolFromSpin()
        #print(problem)
        
        A = problem.Inter
        b = problem.Pot
        c = problem.Const
        #print(b)
        n = len(A)

        vars = []
        for j in range(n):
            vars.append(m.addVar(0, 1, vtype=GRB.CONTINUOUS) )
            #m.addConstr(vars[j]*vars[j] == 1)

        # Populate objective
        obj = QuadExpr()
        for i in range(n):
            for j in range(n):
                if A[i, j] != 0:
                    obj += A[i, j]*vars[i]*vars[j]
        for j in range(n):
            if b[j] != 0:
                obj += b[j]*vars[j]
        obj += c

        m.setObjective(obj, GRB.MAXIMIZE)

        # Solve
        m.optimize()
        
        return m
#------------------------------------------------------------------------------------------------------------------
     
def Laplacian(Adjacency): 
    
    #G = nx.from_numpy_matrix(Adjacency, create_using=nx.MultiGraph)
    #L = csr_matrix.todense(nx.laplacian_matrix(G))
    n = len(Adjacency)
    
    d = np.matmul(np.ones((1, n)), Adjacency)
    d = list(d[0])#d.reshape((n, ))
    print("d",d)
    D = np.diag(d)
    print("D",D)
    
    return D - Adjacency

    #return L


def Adjacency(Laplacian): 
    
    A = copy.deepcopy(Laplacian)
    np.fill_diagonal(A, 0)
    A = -1*A
    
    return A
    
        
#implement bqpjson parsing

def BQPJSON(filename):
    #read from bqpjson 
   
    file = open(filename).read();#open("ran1_b_1.json", "r")
    data = json.loads(file)
    version = data["version"]
    ids = data["id"]
    metadata = data["metadata"]
    variable_ids = data["variable_ids"]
    variable_domain = data["variable_domain"]
    scale = data["scale"]
    offset = data["offset"]
    linear_terms = data["linear_terms"]
    quadratic_terms = data["quadratic_terms"]
    #if haskey(data, "description")
    #description = data["description"]
    #end 
    #if haskey(data, "solutions")
    #solutions = data["solutions"]
    #end

    n = len(variable_ids)
    #transform variable ids to 1:n
    variables = {}
    for i in range(n):
        variables.update({variable_ids[i] : i}) 


    #form matrix A from quadratic terms
    A = np.zeros((n,n), dtype=float)
    for quad_iter in range(len(quadratic_terms)):
        i = variables[quadratic_terms[quad_iter]["id_head"]]
        j = variables[quadratic_terms[quad_iter]["id_tail"]]
        cij = quadratic_terms[quad_iter]["coeff"]
        A[i, j] = -cij/2
        A[j, i] = -cij/2
        
    #form column-vector b from linear terms
    b = np.zeros((n), dtype=float)
    for lin_iter in range (len(linear_terms)):
        i = variables[linear_terms[lin_iter]["id"]]
        h = linear_terms[lin_iter]["coeff"]
        b[i] = -h
    
    # all in all we have maximization    <x, A x> + <b, x> subjecti to boolean/spin constraints
    
    #------------------------------converting to spin problem-----------------------------------
    if (variable_domain == "boolean"):  #Check it!!!
        Inter = A/4
        Pot = b/2 + np.matmul(np.ones(n), A/2)
        c = np.matmul(np.ones(n), b/2)+np.matmul(np.ones(n), np.matmul(A/4, np.ones(n)))
    else:
        Inter = A
        Pot = b
        c = 0.0
    #-------------------------------------------------------------------------------------------    
    
    return (Inter, Pot, c)
        
        
#"""
#-------------------------------------------------------------------
#maximize   <x, Inter x> + <Pot, x> + Const
#subject to spin constraints
#-------------------------------------------------------------------
#"""

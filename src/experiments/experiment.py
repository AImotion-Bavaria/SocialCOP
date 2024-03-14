import numpy as np


class Experiment:
    def __init__(self, n_value, m_value, solvers, repetitions):
        self.n_value = n_value #participants
        self.m_value = m_value #number of goods
        self.solvers = solvers #list of reweighting functions
        self.repetitions = repetitions #number of repetitions

    #https://github.com/oliviaguest/gini/tree/master
    def calculate_gini(self,array):
        array = np.sort(np.array(array)) #cast to sorted numpy array
        index = np.arange(1,array.shape[0]+1) #index per array element
        n = array.shape[0] #number of array elements
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

    def test_gini(self):
    #Test gini calculation 
        a = np.zeros((1000)) #create array of zeros
        a[0] = 1.0 #change first value only to 1.0
        gini_index = self.calculate_gini(a) #as array is not equally distributed, gini should be high
        print(f"Gini-Index Test: Should be close to 1: {gini_index:.4f}\n\n")
        

    def run_experiment(self):
        for solver in self.solvers:
            solved = solver(self.m_value, self.n_value, self.repetitions) #solve with given reweighting function
            gini_index = self.calculate_gini(solved)
            color_circle(gini_index)
            print(f"Gini-Index for {solver.__name__}: {gini_index:.4f}",color_circle(gini_index))
        
def color_circle(value):
    if 0 <= value <0.33:
        return("🟢")
    elif 0.33 <= value <0.66:
        return("🟡") 
    elif 0.66 <= value <=1:
        return("🔴") 


# Run an SQL query and show the results 
# for the comparison between leximin and utilitarian
import json
import os
import sqlite3
import os
import numpy as np

#see https://github.com/oliviaguest/gini
def calculate_gini(array):
    array = np.sort(np.array(array)) #cast to sorted numpy array
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0] #number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

print("scheduling 1, nash", calculate_gini([17, 14, 20]))
print("scheduling 1, util", calculate_gini([19, 12, 20]))
print("project_assignment 4, nash", calculate_gini([4, 5, 5, 4, 4, 5]))
print("project_assignment 4, util", calculate_gini([2, 5, 5, 5, 5, 5]))

# Run an SQL query and show the results 
# for the comparison between leximin and utilitarian
import json
import os

import sqlite3
import os
import numpy as np

def calculate_gini(array):
    array = np.sort(np.array(array)) #cast to sorted numpy array
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0] #number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

# Define the path to the SQLite database file
db_file = os.path.join(os.path.dirname(__file__), 'results_database_leximin_util.db')

# Define the path to the SQL query file
sql_file = os.path.join(os.path.dirname(__file__), 'leximin_utilitarian.sql')

# Read the SQL query from the file
with open(sql_file, 'r') as f:
    sql_query = f.read()

# Execute the SQL query and calculate the geometric mean
with sqlite3.connect(db_file) as conn:
    cursor = conn.cursor()
    cursor.execute(sql_query)
    rows = cursor.fetchall()
    quotients = [row[6] for row in rows]
    leximin_utility_vector=[row[2] for row in rows]
    utilitarian_utility_vector=[row[3] for row in rows]

def print_procentual_change():
    utility_quotients=[]
    for element in range(len(leximin_utility_vector)):
        leximin_utility = json.loads(leximin_utility_vector[element])
        utilitarian_utility = json.loads(utilitarian_utility_vector[element])
        for agent in range(len(leximin_utility)):
            #issue: -20:10 is a lower percentual increase then -10:10
            if(utilitarian_utility[agent]!=0):
                #procentual change is not devined for negative reference values
                utility_quotients.append(((utilitarian_utility[agent])-(leximin_utility[agent]))/abs(utilitarian_utility[agent]))
            else:
                #division by zero not applicable
                utility_quotients.append(np.nan)
        avg = np.average(utility_quotients)
        print("column", element, ":",  utility_quotients, "average: ", avg)
        utility_quotients=[]




gini_results_leximin=[]
gini_results_utilitarian=[]
for element in range(len(leximin_utility_vector)):
    leximin_utility = json.loads(leximin_utility_vector[element])
    utilitarian_utility = json.loads(utilitarian_utility_vector[element])
    gini_results_leximin.append(calculate_gini(leximin_utility))
    gini_results_utilitarian.append(calculate_gini(utilitarian_utility))
    print("Gini at column: ",element,"leximin:",gini_results_leximin[element],"utilitarian:",gini_results_utilitarian[element], "difference:",gini_results_utilitarian[element]-gini_results_leximin[element])  
print("Mean values: difference:", np.average([(b - a) for a, b in zip(gini_results_leximin, gini_results_utilitarian)]), "; average procentual change:", np.average([(b-a)/b if b != 0 else 0 for a, b in zip(gini_results_leximin, gini_results_utilitarian)]))




    

# Calculate the geometric mean of the "quotient" column
#geometric_mean = np.prod(quotients) ** (1 / len(quotients))

# Print the geometric mean
#print("Geometric Mean of 'quotient' column:", geometric_mean)


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

#tables to store both results for every column
gini_results_leximin=[]
gini_results_utilitarian=[]
#for every result
for element in range(len(leximin_utility_vector)):
    leximin_utility = json.loads(leximin_utility_vector[element])
    utilitarian_utility = json.loads(utilitarian_utility_vector[element])
    gini_results_leximin.append(calculate_gini(leximin_utility))
    gini_results_utilitarian.append(calculate_gini(utilitarian_utility))
    print("Gini at column: ",element,"leximin:",gini_results_leximin[element],"utilitarian:",gini_results_utilitarian[element], "difference:",gini_results_utilitarian[element]-gini_results_leximin[element])  
print("Mean values: difference:", np.average([(b - a) for a, b in zip(gini_results_leximin, gini_results_utilitarian)]), "; average procentual change:", np.average([(b-a)/b if b != 0 else 0 for a, b in zip(gini_results_leximin, gini_results_utilitarian)]))



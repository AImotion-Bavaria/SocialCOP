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
db_file = os.path.join(os.path.dirname(__file__), 'results_leximin_utilitarian_rawls.db')
write_file = os.path.join(os.path.dirname(__file__), 'gini_results_leximin_utilitarian_rawls.db')

# Define the path to the SQL query file
sql_file = os.path.join(os.path.dirname(__file__), 'compare_leximin_utilitarian_rawls.sql')

# Read the SQL query from the file
with open(sql_file, 'r') as f:
    sql_query = f.read()

# Execute the SQL query and store results
with sqlite3.connect(db_file) as conn:
    cursor = conn.cursor()
    cursor.execute(sql_query)
    rows = cursor.fetchall()
    leximin_utility_vector=[row[5] for row in rows]
    utilitarian_utility_vector=[row[9] for row in rows]
    rawls_utility_vector=[row[13] for row in rows]

    
    
# Prepare database to store overall result    
if os.path.exists(write_file): 
    os.remove(write_file)  # Delete previous state of resultfile
with sqlite3.connect(write_file) as conn_write: #store sql result in resultfile
        cursor_write = conn_write.cursor()
        cursor_write.execute("CREATE TABLE IF NOT EXISTS results (model TINYTEXT,data_files TINYTEXT,sum_utility_leximin REAL,max_leximin REAL,min_leximin REAL,leximin_utility_vector TINYTEXT,sum_utility_utilitarian REAL,max_utilitarian REAL,min_utilitarian REAL,utilitarian_utility_vector TINYTEXT,sum_utility_rawls REAL,max_rawls REAL,min_rawls REAL,rawls_utility_vector TINYTEXT)")
        for row in rows:
            cursor_write.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
            conn_write.commit()
        cursor_write.execute("ALTER TABLE results ADD COLUMN gini_leximin REAL")
        cursor_write.execute("ALTER TABLE results ADD COLUMN gini_utilitarian REAL")
        cursor_write.execute("ALTER TABLE results ADD COLUMN gini_rawls REAL")

#tables to store both results for every column
gini_results_leximin=[]
gini_results_utilitarian=[]
gini_results_rawls=[]
#for every result
for element in range(len(leximin_utility_vector)):
    leximin_utility = json.loads(leximin_utility_vector[element])
    utilitarian_utility = json.loads(utilitarian_utility_vector[element])
    rawls_utility = json.loads(rawls_utility_vector[element])
    gini_results_leximin.append(calculate_gini(leximin_utility))
    gini_results_utilitarian.append(calculate_gini(utilitarian_utility))
    gini_results_rawls.append(calculate_gini(rawls_utility))
    print("Gini at column: ",element,"leximin:",gini_results_leximin[element],"utilitarian:",gini_results_utilitarian[element], "rawls:", gini_results_rawls[element])
        
    # Append the calculated Gini results to resultfile
    cursor_write.execute("UPDATE results SET gini_leximin = COALESCE(gini_leximin, 0) + ?, \
                    gini_utilitarian = COALESCE(gini_utilitarian, 0) + ?, \
                    gini_rawls = COALESCE(gini_rawls, 0) + ? \
                    WHERE rowid = ?", (gini_results_leximin[element], gini_results_utilitarian[element], gini_results_rawls[element], element+1))
        
    conn_write.commit()

#draw diagram:
import matplotlib.pyplot as plt
data=[gini_results_utilitarian, gini_results_rawls, gini_results_leximin]
labels = ['Utilitarian', 'Rawlsian', 'Leximin']
num_datasets = len(data)

bar_width = 0.2

for i, dataset in enumerate(data):
    plt.bar(np.arange(len(dataset)) + i * bar_width, dataset, bar_width, label=labels[i])

plt.xlabel('Instance', fontsize=12)
plt.ylabel('Gini Index', fontsize=12)
plt.xticks(np.arange(len(data[0])))
plt.legend(fontsize=12)
#plt.show()
plt.savefig('./src/experiments/cluster_results/gini_viz.pdf')


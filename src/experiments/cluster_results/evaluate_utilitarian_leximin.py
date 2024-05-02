# Run an SQL query and show the results 
# for the comparison between leximin and utilitarian
import os

import sqlite3
import os
import numpy as np

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


utility_quotients=[]
import json
for element in range(len(leximin_utility_vector)):
    leximin_utility = json.loads(leximin_utility_vector[element])
    utilitarian_utility = json.loads(utilitarian_utility_vector[element])
    for agent in range(len(leximin_utility)):
        #absolute numbers are required
        utility_quotients.append(abs(utilitarian_utility[agent])/abs(leximin_utility[agent]))
    geometric_mean = np.prod(utility_quotients) ** (1 / len(utility_quotients))
    print("Geometric Mean of column", element, ":",  geometric_mean)
    utility_quotients=[]

# Calculate the geometric mean of the "quotient" column
geometric_mean = np.prod(quotients) ** (1 / len(quotients))

# Print the geometric mean
print("Geometric Mean of 'quotient' column:", geometric_mean)


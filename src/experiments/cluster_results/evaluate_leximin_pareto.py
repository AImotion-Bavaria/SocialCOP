# Run an SQL query and show the results 
# for the comparison between leximin and leximin+pareto
import os

db_file = os.path.join(os.path.dirname(__file__), 'results_database_leximin.db')
raw_data = os.path.join(os.path.dirname(__file__), 'leximin_pareto_speedup_raw.sql')

import sqlite3
import os
import numpy as np

# Define the path to the SQLite database file
db_file = os.path.join(os.path.dirname(__file__), 'results_database_leximin.db')

# Define the path to the SQL query file
sql_file = os.path.join(os.path.dirname(__file__), 'leximin_pareto_speedup_raw.sql')

# Read the SQL query from the file
with open(sql_file, 'r') as f:
    sql_query = f.read()

# Execute the SQL query and calculate the geometric mean
with sqlite3.connect(db_file) as conn:
    cursor = conn.cursor()
    cursor.execute(sql_query)
    rows = cursor.fetchall()
    quotients = [row[4] for row in rows]

# Calculate the geometric mean of the "quotient" column
geometric_mean = np.prod(quotients) ** (1 / len(quotients))

# Print the geometric mean
print("Geometric Mean of 'quotient' column:", geometric_mean)

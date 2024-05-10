# Run an SQL query and show the results 
# for the comparison between leximin and utilitarian
import json
import os
import sqlite3
import os
import numpy as np

db_file = os.path.join(os.path.dirname(__file__), 'gini_results_leximin_utilitarian_rawls.db')
sql_file = os.path.join(os.path.dirname(__file__), 'write_main_table.sql')

# Read the SQL query from the file
with open(sql_file, 'r') as f:
    sql_query = f.read()

# Execute the SQL query and store results
import pandas as pd

with sqlite3.connect(db_file) as conn:
    cursor = conn.cursor()
    cursor.execute(sql_query)
    df = pd.read_sql_query(sql_query, conn)

print(df.head(5))
    
print(df.to_latex(index=False, escape=False))
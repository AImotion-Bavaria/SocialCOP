# Run an SQL query and show the results 
# for the comparison between leximin and utilitarian
import json
import os
import sqlite3
import os
import numpy as np
from scipy.stats import gmean

db_file = os.path.join(os.path.dirname(__file__), 'results_leximin_utilitarian_rawls.db')
sql_file = os.path.join(os.path.dirname(__file__), 'solver_runtime_overhead_raw.sql')
sql_file_preagg = os.path.join(os.path.dirname(__file__), 'solver_runtime_overhead.sql')
# Read the SQL query from the file
with open(sql_file, 'r') as f:
    sql_query = f.read()

with open(sql_file_preagg, 'r') as f:
    sql_query_preagg = f.read()
    
# Execute the SQL query and store results
import pandas as pd

def geommean_df(df:pd.DataFrame):
    return gmean(df.to_numpy().flatten())

with sqlite3.connect(db_file) as conn:
    cursor = conn.cursor()

    # create the user defined function
    conn.create_function("GEOMMEAN", -1, gmean)
    
    cursor.execute(sql_query)
    df = pd.read_sql_query(sql_query, conn)

    # preaggregated
    df_preagg = pd.read_sql_query(sql_query_preagg, conn)

grouped = df.groupby("solver", group_keys=True)[['leximin_overhead']].apply(lambda x: geommean_df(x)  )
print(df_preagg)
df_preagg["GMean Overhead"] = grouped
print(grouped)
print(df_preagg)

print(df_preagg.to_latex(index=False, escape=False, float_format="%.2f"))
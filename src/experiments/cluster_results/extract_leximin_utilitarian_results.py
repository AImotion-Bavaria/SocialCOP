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
df = df.drop(columns="model")    

translations = {
    'bus_tour_1.dzn': 'bus tour 1',
    'bus_tour_3.dzn': 'bus tour 2',
    'bus_tour_2.dzn': 'bus tour 3',
    'bus_tour_4.dzn': 'bus tour 4',
    'bus_tour_5.dzn': 'bus tour 5',
    'generic_1.dzngeneric_1_preferences.dzn': 'tables 1',
    'generic_2.dzngeneric_2_preferences.dzn': 'tables 2',
    'generic_3.dzngeneric_3_preferences.dzn': 'tables 3',
    'generic_4.dzngeneric_4_preferences.dzn': 'tables 4',
    'generic_5.dzngeneric_5_preferences.dzn': 'tables 5',
    'scheduling_1.dzn': 'scheduling 1',
    'scheduling_2.dzn': 'scheduling 2',
    'scheduling_4.dzn': 'scheduling 3',
    'scheduling_5.dzn': 'scheduling 4',
    'scheduling_6.dzn': 'scheduling 5',
    'project_assignment_1.dzn': 'projects 1',
    'project_assignment_2.dzn': 'projects 2',
    'project_assignment_3.dzn': 'projects 3',
    'project_assignment_4.dzn': 'projects 4'
}
df = df.replace({"data_files" : translations})
for gini in ["gini_leximin", "gini_rawls", "gini_utilitarian"]:
    df[gini] = df[gini].round(2)
print(df.to_latex(index=False, escape=False, float_format="%.2f"))
print(df["data_files"].tolist())



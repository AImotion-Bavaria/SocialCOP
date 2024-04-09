import json 

AGENTS_ARRAY = "agents_array"
NUM_AGENTS = "num_agents"
UTILITY_ARRAY = "utility_array"
MAIN_VARIABLES = "main_variables"    
SHARE_FUNCTION = "share_function"  # a MiniZinc function that extracts something that represents the "share" of an agent
# Signature: sf : Agents -> SSp (ShareSpace) 
SHARE_UTIL_AGENT = "share_utility" # a utility that is only defined on what constitutes a share of an agent     
# Signature: su : Agents x SSp -> Int

def read_social_mapping(filename:str) -> dict:
    with open(filename) as file:
        file_contents = file.read()
        parsed_json = json.loads(file_contents)
    return parsed_json
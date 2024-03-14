import json 

AGENTS_ARRAY = "agents_array"
UTILITY_ARRAY = "utility_array"
MAIN_VARIABLES = "main_variables"

def read_social_mapping(filename:str) -> dict:
    with open(filename) as file:
        file_contents = file.read()
        parsed_json = json.loads(file_contents)
    return parsed_json
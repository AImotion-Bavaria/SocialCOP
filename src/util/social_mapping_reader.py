import json 

AGENTS_ARRAY = "agents_array"
NUM_AGENTS = "num_agents"
UTILITY_ARRAY = "utility_array"
MAIN_VARIABLES = "main_variables"   
TIME_SPAN = "time_span" 
SHARE_FUNCTION = "share_function"  # a MiniZinc function that extracts something that represents the "share" of an agent
# Signature: sf : Agents -> SSp (ShareSpace) 
SHARE_UTIL_AGENT = "share_utility" # a utility that is only defined on what constitutes a share of an agent     
# Signature: su : Agents x SSp -> Int

# Mixin keywords that may be used as templates
AGENTS_ARRAY_MIXIN = "AGENTS_ARRAY"
SHARE_UTIL_AGENT_MIXIN = "SHARE_UTIL_AGENT"
SHARE_FUNCTION_MIXIN = "SHARE_FUNCTION"
UTILITY_ARRAY_MIXIN = "UTILITY_ARRAY"


def read_social_mapping(filename:str) -> dict:
    with open(filename) as file:
        file_contents = file.read()
        parsed_json = json.loads(file_contents)
    return parsed_json

def get_substitution_dictionary(social_mapper : dict):
    sub_dict = {AGENTS_ARRAY_MIXIN : social_mapper.get(AGENTS_ARRAY, ""),
                SHARE_FUNCTION_MIXIN : social_mapper.get(SHARE_FUNCTION, ""),
                SHARE_UTIL_AGENT_MIXIN : social_mapper.get(SHARE_UTIL_AGENT, ""),
                UTILITY_ARRAY_MIXIN : social_mapper.get(UTILITY_ARRAY, ""),
                TIME_SPAN : social_mapper.get(TIME_SPAN, 0),
                MAIN_VARIABLES: social_mapper.get(MAIN_VARIABLES, [])}
    return sub_dict
    
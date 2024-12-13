import json 

TRAIN_ARRAY = "train"
TEST_ARRAY = "test"
TRAIN_AGENTS = "train_agent"
TEST_AGENT = "test_agent"
ITERATIONS = "iterations"    
INIT_VALS = "init_vals"  

def read_json_file(filename: str) -> dict: 
    with open(filename) as file:
        file_contents = file.read()
        parsed_json = json.loads(file_contents)
    return parsed_json

#print(read_json_file("C:\\Users\\julia\\OneDrive\\Desktop\\Repositories\\MAPR\\SocialCOP\\src\\stable_baselines\\agents\\test.json"))

def get_substitution_dictionary(social_mapper : dict):
    sub_dict = {TRAIN_ARRAY : social_mapper.get(TRAIN_ARRAY, ""),
                TEST_ARRAY : social_mapper.get(TEST_ARRAY, ""),
                INIT_VALS : social_mapper.get(INIT_VALS, "")}
    
    return sub_dict
#print(get_substitution_dictionary(read_json_file("C:\\Users\\julia\\OneDrive\\Desktop\\Repositories\\MAPR\\SocialCOP\\src\\stable_baselines\\agents\\test.json")))


    
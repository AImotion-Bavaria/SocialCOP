import os 
import logging
from shutil import copyfileobj
debug_dir = None

def create_debug_folder(base_dir):
    global debug_dir
    debug_dir = os.path.join(base_dir, 'debug')
    if not os.path.isdir(debug_dir):
        os.makedirs(debug_dir)
    return debug_dir

def log_and_debug_generated_files(child, debug_prefix, model_counter=0, debug_dir_ = debug_dir):

    with child.files() as files:
        logging.info(files)

        # copy files to a dedicated debug folder
        current_batch_set = set()

        for item in files:
            filename = os.path.basename(item)
            base, extension = os.path.splitext(filename)
            current_file_dest = os.path.join(debug_dir_, f"{model_counter}_{debug_prefix}.{extension}")
            if current_file_dest in current_batch_set:
                file_modifier = "ab"
            else:
                file_modifier = "wb"
                current_batch_set.add(current_file_dest)

            with open(current_file_dest, file_modifier) as destination:
                with open(item, "rb") as source:    
                    copyfileobj(source, destination)
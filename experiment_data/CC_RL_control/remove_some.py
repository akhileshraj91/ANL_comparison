import os
import shutil

DATA_DIR = './'

SOURCE = '/home/cc/compare_control_methods/experiment_data/models_CC_cluster'

protect_list = next(os.walk(SOURCE))[2]
print(protect_list)
files = next(os.walk(DATA_DIR))[1]
for file in files:
    # print("---",file)
    if "preliminaries_stream" in file and os.path.isdir(file):
        PATH_FILE = DATA_DIR+file
        # print(PATH_FILE)
        subfiles = next(os.walk(PATH_FILE))[2]
        for data in subfiles:
            if any(keyword in data for keyword in protect_list):
                print(">>>",data)
                # print(">>>",file) 
            elif "dynamics" in data:
                print("_______________________________________",data)
                print(PATH_FILE)
                # os.rmdir(PATH_FILE)
                shutil.rmtree(PATH_FILE)



import os
import pandas as pd
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
#import yaml
import math
# For data modeling
import scipy.optimize as opt
import numpy as np
import tarfile
yaml = YAML()

experiment_dir = './'

with open(r'./gros_setpoint-100.yaml') as files:
    parameters = yaml.load(files)
    print(parameters)

for cfg in next(os.walk(experiment_dir))[2]:
    if "100" not in cfg and ".yaml" in cfg:
        print(f"_________________{cfg}")
        hyp_ind = cfg.find('-')
        dot_ind = cfg.find('.')
        with open(cfg,'w') as fil:
            #new_params = yaml.load(fil)
            sp = cfg[hyp_ind+1:dot_ind]
            parameters['controller']['setpoint'] = float(sp)/100
            print(parameters)
            yaml.dump(parameters,fil)
            



import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import math
# For data modeling
import scipy.optimize as opt
import numpy as np
import tarfile
from matplotlib import cm
import seaborn as sns
import pickle
# from stable_baselines3 import DDPG
from stable_baselines3 import PPO

import new_code_normalized_Action as ncna
from datetime import datetime
#
#fig, axs = plt.subplots(2)
#fig.suptitle('power and performance against time')

a = {'gros': 0.83, 'dahu': 0.94, 'yeti': 0.89,'CC_control': 0.92}
b = {'gros': 7.07, 'dahu': 0.17, 'yeti': 2.91, 'CC_control': 0.88}
alpha = {'gros': 0.047, 'dahu': 0.032, 'yeti': 0.023, 'CC_control': 0.034}
beta = {'gros': 28.5, 'dahu': 34.8, 'yeti': 33.7, 'CC_control': 29.5}
K_L = {'gros': 25.6, 'dahu': 42.4, 'yeti': 78.5, 'CC_control': 40.1}
tau = 0.33

#
cluster = 'CC_control'
# file1 = open(r'data_dir','rb')
# file2 = open(r'trace_dir','rb')
file1 = open(r'data_dir'+str(cluster),'rb')
file2 = open(r'trace_dir'+str(cluster),'rb')#
data = pickle.load(file1)
traces = pickle.load(file2)
#
#
#
pareto = {}
#
for trace in traces[cluster][0]:
     data[cluster][trace]['aggregated_values']['energy'] = np.nansum([np.nansum([data[cluster][trace]['rapl_sensors']['value'+str(package_number)].iloc[i+1]*data[cluster][trace]['aggregated_values']['rapls_periods'].iloc[i][0] for i in range(0,len(data[cluster][trace]['aggregated_values']['rapls_periods'].index))]) for package_number in range(0,4)])
pareto[cluster] = pd.DataFrame({'Execution Time':[data[cluster][trace]['aggregated_values']['progress_frequency_median']['median'].index[-1] for trace in traces[cluster][0]],'setpoint':[data[cluster][trace]['parameters']['config-file']['controller']['setpoint'] for trace in traces[cluster][0]]},index=[data[cluster][trace]['aggregated_values']['energy']/10**3 for trace in traces[cluster][0]])
pareto[cluster].sort_index(inplace=True)
#
#
execution_time_power = pareto[cluster].index*pareto[cluster]['Execution Time']
#
#
# # FIGURE 7
cmap = cm.get_cmap('viridis')
#
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.6,6.6))
cb = axes.scatter(pareto[cluster].index,pareto[cluster]['Execution Time'], marker='.', c=(1-pareto[cluster]['setpoint']), cmap=cmap, s=30)

# #plt.show()
#plt.colorbar(cb,label='Degradation $\epsilon$ [unitless]')
axes.grid(True)
axes.set_ylabel('Execution time [s]')
axes.set_xlabel('Energy consumption [kJ]')

# fig2, axes2= plt.subplots(nrows=1, ncols=1, figsize=(6.6,6.6))
# fb = axes2.scatter(range(len(execution_time_power)),execution_time_power,marker='.', c=(1-pareto[cluster]['setpoint']), cmap=cmap, s=30)
#
#
# plt.colorbar(fb,label='Energy')
# axes2.grid(True)
# axes2.set_ylabel('Energy')
# axes2.set_xlabel('sample number')



# experiment_dir = str(cluster)+'/'
experiment_dir = './experiment_data/models_CC_cluster/'
clusters = next(os.walk(experiment_dir))[2] # clusters are name of folders

sample_no = 549
# optimal_mat = []
# dyn_c = 0
for dynamics in clusters:
    if "dynamics" in dynamics:
        ind1 = dynamics.find('_')
        ind2 = dynamics.find('___')
        # print(ind1,ind2,ind3)
        c_1 = round(float(dynamics[ind1+1:ind2]),2)
        c_2 = round(float(dynamics[ind2+3:]),2)
        # print(dynamics)
        model = PPO.load(experiment_dir+dynamics)
        iterations = [10000]

        for et in iterations:
            count = 0
            dones = False
            env = ncna.Custom_env(et)
            obs = env.observation_space.sample()
            energy = 0
            actions = []
            performances = []
            while not dones:
                action, _states = model.predict(obs,deterministic=True)
                obs, rewards, dones, info = env.step(action)
                performances.append(ncna.abnormal_obs(obs))
                ab_action = ncna.abnormal_action(action)
                consumed_power = a[cluster]*ab_action+b[cluster]
                # energy += ncna.abnormal_action(action)
                energy += consumed_power
                actions.append(ncna.abnormal_action(action))
                obs = obs.reshape((1,))
                count += 1

            optimal_val = energy/1000
            # optimal_mat.append(optimal_val)
            axes.scatter(optimal_val,count,c='r', s=10, marker='*')
 #           plt.text(optimal_val,count,(str(c_1)+","+str(c_2)))
            sample_no += 1
            plt.draw()
        # dyn_c+=1
        # if dyn_c > 10:
        #     break

now = datetime.now()


plt.savefig("./figures_normal/result_"+str(now)+".png")

plt.show()

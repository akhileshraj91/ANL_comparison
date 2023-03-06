import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import cm
import warnings
import datetime
import statistics as stat
import traceback

warnings.simplefilter(action='ignore', category=FutureWarning)

experiment_type = 'controller'
DIR = './experiment_results/'

folders = os.listdir(DIR)
final_list = folders

ET_list = []
E_list = []
ET_list_minimum = []
E_list_minimum = []
ET_list_maximum = []
E_list_maximum = []


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


colors = get_cmap(len(final_list))

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.6, 6.6))
fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(6.6, 6.6))
# fig_label, axes_label = plt.subplots()
# axes_label.axis(False)


data = {}
trace = 'sample_data'
color_index = 0
flag_optimal = 0
flag_minimal = 0
flag_maximal = 0
for folds in final_list:
    try:
        legend_name = f'{folds} PCAP'
        experiment_dir = DIR + folds + '/'
        # print(experiment_dir)
        cluster = 'gros'
        traces = {}
        traces_tmp = {}

        traces[cluster] = pd.DataFrame()

        try:
            traces[cluster] = next(os.walk(experiment_dir))[2]
        except Exception as e:
            print(e)

        data = {}
        data[cluster] = {}
        trace = 'sample_data'

        pubMeasurements = pd.read_csv(experiment_dir + "dump_pubMeasurements.csv")
        pubProgress = pd.read_csv(experiment_dir + "dump_pubProgress.csv")

        print(f"{ folds }'s  Execution Time is: ",pubProgress.values[-1][0] - pubProgress.values[0][0])
        progress_sensor = pd.DataFrame({'timestamp': pubProgress['msg.timestamp'], 'value': pubProgress['sensor.value']})
    except:
        traceback.print_exc()

# res_ET = stat.pstdev(ET_list)
# print("Standard Deviation in Execution Time is:", res_ET)
# res_E = stat.pstdev(E_list)
# print("Standar Deviation in Energy Consumption is:", res_E)
# axes.scatter(stat.mean(E_list), stat.mean(ET_list), marker='*', c='k', s=20, label='MEAN VALUE - OPTIMAL PCAP')
#
# res_ET_minimum = stat.pstdev(ET_list_minimum)
# print("Standard Deviation in Execution Time for minimum control is:", res_ET_minimum)
# res_E_minimum = stat.pstdev(E_list_minimum)
# print("Standar Deviation in Energy Consumption for minimum control is:", res_E_minimum)
# axes.scatter(stat.mean(E_list_minimum), stat.mean(ET_list_minimum), marker='*', c='k', s=20,
#              label='MEAN VALUE - MINIMAL PCAP')
#
# res_ET_maximum = stat.pstdev(ET_list_maximum)
# print("Standard Deviation in Execution Time for maximum control is:", res_ET_maximum)
# res_E_maximum = stat.pstdev(E_list_maximum)
# print("Standar Deviation in Energy Consumption for maximum control is:", res_E_maximum)
# axes.scatter(stat.mean(E_list_maximum), stat.mean(ET_list_maximum), marker='*', c='k', s=20,
#              label='MEAN VALUE - MAXIMAL PCAP')
#
# # label_params = axes.get_legend_handles_labels()
# # axes_label.legend(*label_params, loc="center")
#
# axes.grid(True)
# axes.set_ylabel('Execution time [s]')
# axes.set_xlabel('Energy consumption [kJ]')
# leg = axes.legend()
# axes.set_title('Plot of total energy consumed against execution time for 100 executions')
#
# axes2.grid(True)
# axes2.set_xlabel('Time [s]')
# axes2.set_ylabel('Instantaneous power sensed by RAPL sensors')
# leg2 = axes2.legend()
# axes2.set_title('Plot of control action taken against time steps for 100 executions')
#
# results_dir = './Results/'
# if not os.path.exists(results_dir):
#     os.makedirs(f'{results_dir}')
# now = datetime.datetime.now()
# fig.savefig(results_dir + '_' + str(now) + '_ENERGY_EXECUTION.pdf')
# fig2.savefig(results_dir + '_' + str(now) + '_PCAP_TIME_STEPS.pdf')
# # fig_label.savefig(results_dir+'_'+str(now)+'_LABEL_ONLY.pdf',bbox_inches='tight')
#
# plt.show()

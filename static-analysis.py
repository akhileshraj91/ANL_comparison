#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:27:26 2020

@author: sophiecerf
"""

# Libraries
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
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Experiment selection and load data
# =============================================================================
# Getting the right paths
exp_type = 'static_characteristic' # ex: 'stairs' 'identification' 'static_characteristic' 'controller' XXX
#experiment_dir = '/home/cc/europar-96-artifacts/dataset/'+exp_type+'/experiments-data/' # XXX
#print(experiment_dir,"...")
experiment_dir = './experiment_data/'
#clusters = next(os.walk(experiment_dir))[1] # clusters are name of folders
clusters = ['CC_identification']
print(clusters)
if (exp_type == 'stairs') or (exp_type == 'static_characteristic'):
    experiment_type = 'identification'
else:
    experiment_type = exp_type


traces = {} 
traces_tmp = {}
for cluster in clusters:
    traces[cluster] = pd.DataFrame()
    print(cluster,"...")

    if next(os.walk(experiment_dir+cluster))[1] == []:
        files = os.listdir(experiment_dir+cluster)
        for fname in files:
            if fname.endswith("tar.xz"):
                tar = tarfile.open(experiment_dir+cluster+'/'+fname, "r:xz") 
                tar.extractall(path=experiment_dir+cluster+'/'+fname[:-7])
                tar.close()
    traces[cluster][0] = next(os.walk(experiment_dir+cluster))[1] 
#print("__________",traces[cluster][0])

# Processing data format to dataframe
data = {}
for cluster in clusters:
    data[cluster] = {}
    for trace in traces[cluster][0]:
        #print("_____",trace)
        data[cluster][trace] = {}
        folder_path = experiment_dir+cluster+'/'+trace 
        #print(f"________________{folder_path}")
        # Trace experimental plan: parameters or log
        #if os.path.isfile(folder_path+'/SUCCESS'):
         #   data[cluster][trace]['SUCCESS'] = True
        #else:
         #   data[cluster][trace]['SUCCESS'] = False
        if os.path.isfile(folder_path+'/parameters.yaml'):
            with open(folder_path+"/parameters.yaml") as file:
                data[cluster][trace]['parameters'] = yaml.load(file, Loader=yaml.FullLoader)
                with open(folder_path+'/'+data[cluster][trace]['parameters']['config-file']) as file:
                    data[cluster][trace]['parameters']['config-file'] = yaml.load(file, Loader=yaml.FullLoader)
        data[cluster][trace]['identification-runner-log'] = pd.read_csv(folder_path+"/"+experiment_type+"-runner.log", sep = '\0', names = ['created','levelname','process','funcName','message'])
        data[cluster][trace]['enforce_powercap'] = data[cluster][trace]['identification-runner-log'][data[cluster][trace]['identification-runner-log']['funcName'] == 'enforce_powercap']
        data[cluster][trace]['enforce_powercap'] = data[cluster][trace]['enforce_powercap'].set_index('created')
        data[cluster][trace]['enforce_powercap']['powercap'] = [''.join(c for c in data[cluster][trace]['enforce_powercap']['message'][i] if c.isdigit()) for i in data[cluster][trace]['enforce_powercap'].index]
        # Loading sensors data files
        pubMeasurements = pd.read_csv(folder_path+"/dump_pubMeasurements.csv")
        pubProgress = pd.read_csv(folder_path+"/dump_pubProgress.csv")
         # Extracting sensor data
        rapl_sensor0 = rapl_sensor1 = rapl_sensor2 = rapl_sensor3 = downstream_sensor = pd.DataFrame({'timestamp':[],'value':[]})
        for i, row in pubMeasurements.iterrows():
            if row['sensor.id'] == 'RaplKey (PackageID 0)':
                rapl_sensor0 = rapl_sensor0.append({'timestamp':row['sensor.timestamp'],'value':row['sensor.value']}, ignore_index=True)
            elif row['sensor.id'] == 'RaplKey (PackageID 1)':
                rapl_sensor1 = rapl_sensor1.append({'timestamp':row['sensor.timestamp'],'value':row['sensor.value']}, ignore_index=True)
            elif row['sensor.id'] == 'RaplKey (PackageID 2)':
                rapl_sensor2 = rapl_sensor1.append({'timestamp':row['sensor.timestamp'],'value':row['sensor.value']}, ignore_index=True)
            elif row['sensor.id'] == 'RaplKey (PackageID 3)':
                rapl_sensor3 = rapl_sensor1.append({'timestamp':row['sensor.timestamp'],'value':row['sensor.value']}, ignore_index=True)
        progress_sensor = pd.DataFrame({'timestamp':pubProgress['msg.timestamp'],'value':pubProgress['sensor.value']})
        # Writing in data dict
        data[cluster][trace]['rapl_sensors'] = pd.DataFrame({'timestamp':rapl_sensor0['timestamp'],'value0':rapl_sensor0['value'],'value1':rapl_sensor1['value'],'value2':rapl_sensor2['value'],'value3':rapl_sensor3['value']})
        data[cluster][trace]['performance_sensors'] = pd.DataFrame({'timestamp':progress_sensor['timestamp'],'progress':progress_sensor['value']})
        #data[cluster][trace]['nrm_downstream_sensors'] = pd.DataFrame({'timestamp':downstream_sensor['timestamp'],'downstream':downstream_sensor['value']})
        # Indexing on elasped time since the first data point
        data[cluster][trace]['first_sensor_point'] = min(data[cluster][trace]['rapl_sensors']['timestamp'][0], data[cluster][trace]['performance_sensors']['timestamp'][0])#, data[cluster][trace]['nrm_downstream_sensors']['timestamp'][0])
        data[cluster][trace]['rapl_sensors']['elapsed_time'] = (data[cluster][trace]['rapl_sensors']['timestamp'] - data[cluster][trace]['first_sensor_point'])
        data[cluster][trace]['rapl_sensors'] = data[cluster][trace]['rapl_sensors'].set_index('elapsed_time')
        data[cluster][trace]['performance_sensors']['elapsed_time'] = (data[cluster][trace]['performance_sensors']['timestamp'] - data[cluster][trace]['first_sensor_point'])
        data[cluster][trace]['performance_sensors'] = data[cluster][trace]['performance_sensors'].set_index('elapsed_time')

# Compute extra metrics: averages, frequencies, upsampling
for cluster in clusters:
    for trace in traces[cluster][0]:
        # Average sensors value
        avg0 = data[cluster][trace]['rapl_sensors']['value0'].mean()
        avg1 = data[cluster][trace]['rapl_sensors']['value1'].mean()
        avg2 = data[cluster][trace]['rapl_sensors']['value2'].mean()
        avg3 = data[cluster][trace]['rapl_sensors']['value3'].mean()
        data[cluster][trace]['aggregated_values'] = {'rapl0':avg0,'rapl1':avg1,'rapl2':avg2,'rapl3':avg3,'progress':data[cluster][trace]['performance_sensors']['progress']}#'rapl0_std':std0,'rapl1_std':std1,'rapl2_std':std2,'rapl3_std':std3,'downstream':data[cluster][trace]['nrm_downstream_sensors']['downstream'].mean(),'progress':data[cluster][trace]['performance_sensors']['progress']}
        avgs = pd.DataFrame({'averages':[avg0, avg1, avg2, avg3]})
        data[cluster][trace]['aggregated_values']['rapls'] = avgs.mean()[0]
        # Sensors periods and frequencies
            # RAPL
        rapl_elapsed_time = data[cluster][trace]['rapl_sensors'].index
        data[cluster][trace]['aggregated_values']['rapls_periods'] = pd.DataFrame([rapl_elapsed_time[t]-rapl_elapsed_time[t-1] for t in range(1,len(rapl_elapsed_time))], index=[rapl_elapsed_time[t] for t in range(1,len(rapl_elapsed_time))], columns=['periods'])
            # Progress
        performance_elapsed_time = data[cluster][trace]['performance_sensors'].index
        data[cluster][trace]['aggregated_values']['performance_frequency'] = pd.DataFrame([1/(performance_elapsed_time[t]-performance_elapsed_time[t-1]) for t in range(1,len(performance_elapsed_time))], index=[performance_elapsed_time[t] for t in range(1,len(performance_elapsed_time))], columns=['frequency'])
        # Execution time:
        data[cluster][trace]['aggregated_values']['execution_time'] = performance_elapsed_time[-1]
        data[cluster][trace]['aggregated_values']['upsampled_timestamps'] = data[cluster][trace]['rapl_sensors'].index
        # Computing count and frequency at upsampled_frequency:
        data[cluster][trace]['aggregated_values']['progress_frequency_median'] = pd.DataFrame({'median':np.nanmedian(data[cluster][trace]['aggregated_values']['performance_frequency']['frequency'].where(data[cluster][trace]['aggregated_values']['performance_frequency'].index<= data[cluster][trace]['aggregated_values']['upsampled_timestamps'][0],0)),'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][0]}, index=[0])
        idx = 0  # index of powercap change in log
        data[cluster][trace]['aggregated_values']['pcap'] = pd.DataFrame({'pcap':math.nan,'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][0]}, index=[0])
        for t in range(1,len(data[cluster][trace]['aggregated_values']['upsampled_timestamps'])):
             data[cluster][trace]['aggregated_values']['progress_frequency_median'] = data[cluster][trace]['aggregated_values']['progress_frequency_median'].append({'median':np.nanmedian(data[cluster][trace]['aggregated_values']['performance_frequency']['frequency'].where((data[cluster][trace]['aggregated_values']['performance_frequency'].index>= data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t-1]) & (data[cluster][trace]['aggregated_values']['performance_frequency'].index <=data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]))),'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]}, ignore_index=True)
             if (experiment_type == 'controller') or (experiment_type == 'identification'): 
                 if (data[cluster][trace]['enforce_powercap'].index[idx]-data[cluster][trace]['first_sensor_point'])<data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]:
                     if idx < len(data[cluster][trace]['enforce_powercap'])-1:           
                         idx = idx +1
                 if (data[cluster][trace]['enforce_powercap'].index[0]-data[cluster][trace]['first_sensor_point'])>data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]:
                    data[cluster][trace]['aggregated_values']['pcap'] = data[cluster][trace]['aggregated_values']['pcap'].append({'pcap':math.nan,'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]}, ignore_index=True)
                 elif (data[cluster][trace]['enforce_powercap'].index[-1]-data[cluster][trace]['first_sensor_point'])<data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]:
                     data[cluster][trace]['aggregated_values']['pcap'] = data[cluster][trace]['aggregated_values']['pcap'].append({'pcap':int(data[cluster][trace]['enforce_powercap']['powercap'].iloc[-1]),'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]}, ignore_index=True)
                 else:
                     data[cluster][trace]['aggregated_values']['pcap'] = data[cluster][trace]['aggregated_values']['pcap'].append({'pcap':int(data[cluster][trace]['enforce_powercap']['powercap'].iloc[idx-1]),'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]}, ignore_index=True)
        data[cluster][trace]['aggregated_values']['pcap'] = data[cluster][trace]['aggregated_values']['pcap'].set_index('timestamp')
        if (experiment_type == 'preliminaries') or (experiment_type == 'static_characteristic'):
            print("cheking execution of this line.")
            data[cluster][trace]['aggregated_values']['pcap'] = pd.DataFrame({'pcap':int(data[cluster][trace]['parameters']['powercap']),'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][0]}, index=[0])
            for t in range(1,len(data[cluster][trace]['aggregated_values']['upsampled_timestamps'])):
                data[cluster][trace]['aggregated_values']['pcap'] = data[cluster][trace]['aggregated_values']['pcap'].append({'pcap':int(data[cluster][trace]['parameters']['powercap']),'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]}, ignore_index=True)
            data[cluster][trace]['aggregated_values']['pcap'] = data[cluster][trace]['aggregated_values']['pcap'].set_index('timestamp')
        data[cluster][trace]['aggregated_values']['progress_frequency_median'] = data[cluster][trace]['aggregated_values']['progress_frequency_median'].set_index('timestamp')


#print(f"________{data[cluster][trace]}__")
# =============================================================================
# STAIRS
# =============================================================================

# FIGURE 3
#print(f"The parameters in the content is {data[cluster][trace]['parameters']}")
for cluster in clusters:
    for my_trace in traces[cluster][0]:
        x_zoom = [0,100]#[0,len(data[cluster][my_trace]['aggregated_values']['progress_frequency_median']['median'])]
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5.7,6.6))
        data[cluster][my_trace]['aggregated_values']['progress_frequency_median']['median'].plot(color='k',ax=axes[0], marker='o', markersize=3,linestyle='')
        axes[0].set_ylabel('Progress [Hz]')
        axes[0].set_xlabel('')
        axes[0].legend(['Measure','Objective value','Objective value ±5%'],fontsize='small')
        axes[0].set_xlim(x_zoom)
        axes[0].grid(True)
        data[cluster][my_trace]['aggregated_values']['pcap'].plot(color='k',ax=axes[1], style=".")#, style="+",  markersize=4)
        data[cluster][my_trace]['rapl_sensors']['value0'].plot(color='lightcoral',ax=axes[1], marker="+", linestyle='')#, style="+",  markersize=4)
        data[cluster][my_trace]['rapl_sensors']['value1'].plot(color='lightcoral',ax=axes[1], marker="+", linestyle='')
        data[cluster][my_trace]['rapl_sensors']['value2'].plot(color='lightcoral',ax=axes[1], marker="+", linestyle='')#, style="+",  markersize=4)
        data[cluster][my_trace]['rapl_sensors']['value3'].plot(color='lightcoral',ax=axes[1], marker="+", linestyle='')#, style="+",  markersize=4)
        axes[1].set_ylabel('Power [W]')
        axes[1].set_xlabel('Time [s]')
        axes[1].legend(['Powercap','Measure'],fontsize='small',ncol=1) # ,'Measure - package1'
        axes[1].grid(True)
        axes[1].set_xlim(x_zoom)

plt.savefig('fig_3.pdf')
# =============================================================================
# STATIC CHARACTERISTIC
# =============================================================================

# Getting parameters a & b 

# init values
pmin = 40
pmax = 120
power_parameters0 = [1, 0]                                        


# Reshaping data: 
prequestedvsmeasured = {}
for cluster in clusters:
    prequestedvsmeasured[cluster] = pd.DataFrame()
#    print("___",[data[cluster][trace]['parameters']['config-file']['actions'][0]['args'][0] for trace in traces[cluster][0]])
    prequestedvsmeasured[cluster]['requested_pcap'] = [data[cluster][trace]['parameters']['config-file']['actions'][0]['args'][0] for trace in traces[cluster][0]]
    prequestedvsmeasured[cluster]['0'] = [data[cluster][trace]['aggregated_values']['rapl0'] for trace in traces[cluster][0]]
    prequestedvsmeasured[cluster]['1'] = [data[cluster][trace]['aggregated_values']['rapl1'] for trace in traces[cluster][0]]
    prequestedvsmeasured[cluster]['2'] = [data[cluster][trace]['aggregated_values']['rapl2'] for trace in traces[cluster][0]]
    prequestedvsmeasured[cluster]['3'] = [data[cluster][trace]['aggregated_values']['rapl3'] for trace in traces[cluster][0]]
    prequestedvsmeasured[cluster]['rapls'] = [data[cluster][trace]['aggregated_values']['rapls'] for trace in traces[cluster][0]]
    prequestedvsmeasured[cluster]['pcap_requested'] = prequestedvsmeasured[cluster]['requested_pcap']
    prequestedvsmeasured[cluster] = prequestedvsmeasured[cluster].set_index('requested_pcap')
    prequestedvsmeasured[cluster].sort_index(inplace=True)
  


print("____________",prequestedvsmeasured[cluster])



#Powercap to Power measure model:
def powermodel(power_requested, slope, offset):
    return slope*power_requested+offset

# Optimizing power model parameters
power_model_data = {}
power_model = {}
power_parameters = {}
r_squared_power_actuator = {}
for cluster in clusters:
    # optimized params
    power_parameters[cluster], power_parameters_cov = opt.curve_fit(powermodel, prequestedvsmeasured[cluster]['pcap_requested'], prequestedvsmeasured[cluster]['rapls'], p0=power_parameters0)     # /!\ model computed with package 0
    # Model
    power_model[cluster] = powermodel(prequestedvsmeasured[cluster]['pcap_requested'].loc[pmin:pmax], power_parameters[cluster][0], power_parameters[cluster][1]) # model with fixed alpha

# Getting K_L, alpha, beta
sc = {}
sc_requested = {}
pcap2perf_model = {}
power2perf_params = {}

elected_performance_sensor = 'progress_frequency_median' # choose between: 'average_performance_periods' 'average_progress_count' 'average_performance_frequency'
for cluster in clusters:
    sc[cluster] = pd.DataFrame([data[cluster][trace]['aggregated_values'][elected_performance_sensor]['median'].mean() for trace in traces[cluster][0]], index=[data[cluster][trace]['aggregated_values']['rapls'] for trace in traces[cluster][0]], columns=[elected_performance_sensor])
    sc[cluster].sort_index(inplace=True)
    sc_requested[cluster] = pd.DataFrame([data[cluster][trace]['aggregated_values'][elected_performance_sensor]['median'].mean() for trace in traces[cluster][0]], index=[data[cluster][trace]['parameters']['config-file']['actions'][0]['args'][0] for trace in traces[cluster][0]], columns=[elected_performance_sensor])
    sc_requested[cluster].sort_index(inplace=True)

def power2perf(power, alpha, perf_inf, power_0): # general model formulation
    return perf_inf*(1-np.exp(-alpha*(power-power_0)))

def pcap2perf(pcap, a, b, perf_inf, alpha, power_0): # general model formulation
    return perf_inf*(1-np.exp(-alpha*(a*pcap+b-power_0)))

# Model optimisation 
for cluster in clusters:
    # init param
    power2perf_param0 = [0.04, (sc[cluster].at[sc[cluster].index[-1],elected_performance_sensor]+sc[cluster].at[sc[cluster].index[-2],elected_performance_sensor]+sc[cluster].at[sc[cluster].index[-3],elected_performance_sensor])/3, min(sc[cluster].index)]                                        # guessed params
    # Optimization
    power2perf_param_opt, power2perf_param_cov = opt.curve_fit(power2perf, sc[cluster].index, sc[cluster][elected_performance_sensor], p0=power2perf_param0)     
    power2perf_params[cluster] = power2perf_param_opt
    # Model
    pcap2perf_model[cluster] = pcap2perf(sc_requested[cluster].index, power_parameters[cluster][0], power_parameters[cluster][1], power2perf_params[cluster][1], power2perf_params[cluster][0], power2perf_params[cluster][2]) # model with optimized perfinf

# plot style
clusters_styles = {'yeti':'orange','gros':'black','dahu':'skyblue','CC_identification':'black'}
clusters_markers = {'yeti':'o','gros':'x','dahu':'v','CC_identification':'x'}
plt.rcParams.update({'font.size': 14})

# FIGURE 4a
fig, axes = plt.subplots(nrows = 2, ncols=1, figsize=(6.6,6.6))
# fig, axes = plt.subplots(nrows = 2, ncols=1, figsize=(6.6,6.6), gridspec_kw={'top':1.25,'bottom':1,'left':1,'right':1.5})
margin = 0.3937  # 1 cm in inches
top_margin = 1 * margin / fig.get_figheight()
bottom_margin = 1.5 * margin / fig.get_figheight()
left_margin = 2.5 * margin / fig.get_figwidth()
right_margin = 1 * margin / fig.get_figwidth()

fig.subplots_adjust(
    top=1-top_margin,
    bottom=bottom_margin,
    left=left_margin,
    right=1-right_margin,
    hspace=0.3,
    wspace=0.4
)
# axes_1[0].plot()
# axes_1[1].plot()
for cluster in clusters:
    # Pcap
    sc_requested[cluster][elected_performance_sensor].plot(ax=axes[0],color='red',marker='*',linestyle='') # pcap vs measured progress
    axes[0].plot(sc_requested[cluster].index,pcap2perf_model[cluster],color='black') # pcap vs. modelled progress
axes[0].grid(True)
axes[0].set_ylabel('Progress [Hz]')
axes[0].set_xlabel('Powercap [W]')
axes[0].set_ylim([0,80])
legend = []
for cluster in clusters:
    legend += ['cluster: '+' - measures']
    legend += ['cluster: '+' - model']
axes[0].legend(legend,fontsize='x-small',loc='upper right',ncol=1)
# fig_1.savefig('4a.pdf')

# FIGURE 4b
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.6,6.6))
for cluster in clusters:
    # Linear Pcap
    axes[1].plot(-np.exp(-power2perf_params[cluster][0]*(power_parameters[cluster][0]*sc_requested[cluster].index+power_parameters[cluster][1]-power2perf_params[cluster][2])),sc_requested[cluster][elected_performance_sensor]- power2perf_params[cluster][1],color='red', marker=clusters_markers[cluster],linestyle='') # data (lin with fixed alpha = 0.04)
    axes[1].plot(-np.exp(-power2perf_params[cluster][0]*(power_parameters[cluster][0]*sc_requested[cluster].index+power_parameters[cluster][1]-power2perf_params[cluster][2])),pcap2perf_model[cluster]-power2perf_params[cluster][1],color='black') # model 0.04
axes[1].grid(True)
axes[1].set_ylabel('Linearized Progress [Hz]')
axes[1].set_xlabel('Linearized Powercap [unitless]')
legend = []
for cluster in clusters:
    legend += ['cluster: '+' - measures']
    legend += ['cluster: '+' - model']
axes[1].legend(legend,fontsize='x-small',loc='lower right',ncol=1)

# plt.subplots_adjust(bottom=1cm, top=1cm)

fig.savefig('4ab.pdf')

print("We are now going to print the tables.")
# Table 2
for cluster in clusters:
    print(cluster)
    print('RAPL slope - a - [1]: '+str(round(power_parameters[cluster][0],2)))
    print('RAPL offset - b - [W]: '+str(round(power_parameters[cluster][1],2)))
    print('α - [1/W]: '+str(round(power2perf_params[cluster][0],3)))
    print('power offset - β - [W]: '+str(round(power2perf_params[cluster][2],1)))
    print('linear gain - K_L - [Hz]]: '+str(round(power2perf_params[cluster][1],1)))


with open(r'./experiment_inputs/control_SP/gros_setpoint-53.yaml') as file:
    parameters = yaml.full_load(file)
    print(type(parameters), parameters)
    parameters['rapl']['slope'] = str(round(power_parameters[cluster][0],2))
    parameters['rapl']['offset'] = str(round(power_parameters[cluster][1],2))
    parameters['model']['alpha'] = str(round(power2perf_params[cluster][0],3))
    parameters['model']['beta'] = str(round(power2perf_params[cluster][2],1))
    parameters['model']['gain'] = str(round(power2perf_params[cluster][1],1))
    print(parameters)

# # parameters = pd.DataFrame()
# # parameters['version'] = 1
# # parameters['rapl'] = pd.DataFrame()
# # parameters['model'] = pd.DataFrame()
# # parameters['controller'] = pd.DataFrame()
# # parameters['rapl']['slope'] = round(power_parameters[cluster][0],2)
# # parameters['rapl']['offset'] = round(power_parameters[cluster][1],2)
# # parameters['model']['alpha'] = round(power2perf_params[cluster][0],3)
# # parameters['model']['beta'] = round(power2perf_params[cluster][2],1)
# # parameters['model']['gain'] = round(power2perf_params[cluster][1],1)
# # parameters['model']['time-constant'] = round(0.33)
# # parameters['controller']['setpoint'] = round(0.99)
# # parameters['controller']['response-time'] = round(30)
# # parameters['controller']['power-range'] = [40, 120]
#
# params = pd.DataFrame()
# params.insert('version',1)
# params.insert('rapl',['slope','offset'],[1,2])



with open(r'./experiment_data/params.yaml','w') as file2:
    yaml.dump(parameters, file2)

    
# plt.show()    

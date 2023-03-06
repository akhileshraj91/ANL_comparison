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


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.6,6.6))
fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(6.6,6.6))
#fig_label, axes_label = plt.subplots()
#axes_label.axis(False)


data = {}
trace = 'sample_data'
color_index = 0
flag_optimal = 0
flag_minimal = 0
flag_maximal = 0
total_count = 0
for folds in final_list:
    try:
        legend_name = f'{folds} PCAP'
        experiment_dir = DIR+folds+'/'
        # print(experiment_dir)
        cluster = 'gros'
        traces = {}
        traces_tmp = {}

        traces[cluster] = pd.DataFrame()
        
        try:
            traces[cluster] = next(os.walk(experiment_dir))[2]
        except Exception as e: print(e)
            


        data = {}
        data[cluster] = {}
        trace = 'sample_data'


        pubMeasurements = pd.read_csv(experiment_dir+"dump_pubMeasurements.csv")
        pubProgress = pd.read_csv(experiment_dir + "dump_pubProgress.csv")

        data[cluster][trace] = {}
        rapl_sensor0 = rapl_sensor1 = downstream_sensor = pd.DataFrame(
                {'timestamp': [], 'value': []})
        for i, row in pubMeasurements.iterrows():
            if row['sensor.id'] == 'RaplKey (PackageID 0)':
                rapl_sensor0 = rapl_sensor0.append({'timestamp': row['sensor.timestamp'], 'value': row['sensor.value']},ignore_index=True)
            elif row['sensor.id'] == 'RaplKey (PackageID 1)':
                rapl_sensor1 = rapl_sensor1.append({'timestamp': row['sensor.timestamp'], 'value': row['sensor.value']},ignore_index=True)

        data[cluster][trace]['rapl_sensors'] = pd.DataFrame({'timestamp': rapl_sensor0['timestamp'], 'value0': rapl_sensor0['value'], 'value1': rapl_sensor1['value']})


        progress_sensor = pd.DataFrame({'timestamp': pubProgress['msg.timestamp'], 'value': pubProgress['sensor.value']})

        data[cluster][trace]['performance_sensors'] = pd.DataFrame({'timestamp': progress_sensor['timestamp'], 'progress': progress_sensor['value']})


        data[cluster][trace]['first_sensor_point'] = min(data[cluster][trace]['rapl_sensors']['timestamp'][0],        data[cluster][trace]['performance_sensors']['timestamp'][0])
        data[cluster][trace]['rapl_sensors']['elapsed_time'] = (data[cluster][trace]['rapl_sensors']['timestamp'] - data[cluster][trace]['first_sensor_point'])
        data[cluster][trace]['rapl_sensors'] = data[cluster][trace]['rapl_sensors'].set_index('elapsed_time')
        data[cluster][trace]['performance_sensors']['elapsed_time'] = (
                        data[cluster][trace]['performance_sensors']['timestamp'] - data[cluster][trace][
                    'first_sensor_point'])
        data[cluster][trace]['performance_sensors'] = data[cluster][trace]['performance_sensors'].set_index(
                'elapsed_time')

        avg0 = data[cluster][trace]['rapl_sensors']['value0'].mean()
        avg1 = data[cluster][trace]['rapl_sensors']['value1'].mean()

        data[cluster][trace]['aggregated_values'] = {'rapl0': avg0, 'rapl1': avg1,
                                                     'progress': data[cluster][trace]['performance_sensors'][
                                                         'progress']} 
        avgs = pd.DataFrame({'averages': [avg0, avg1]})
        data[cluster][trace]['aggregated_values']['rapls'] = avgs.mean()[0]
        rapl_elapsed_time = data[cluster][trace]['rapl_sensors'].index
        data[cluster][trace]['aggregated_values']['rapls_periods'] = pd.DataFrame(
            [rapl_elapsed_time[t] - rapl_elapsed_time[t - 1] for t in range(1, len(rapl_elapsed_time))],
            index=[rapl_elapsed_time[t] for t in range(1, len(rapl_elapsed_time))], columns=['periods'])
        performance_elapsed_time = data[cluster][trace]['performance_sensors'].index
        data[cluster][trace]['aggregated_values']['performance_frequency'] = pd.DataFrame(
            [1 / (performance_elapsed_time[t] - performance_elapsed_time[t - 1]) for t in
             range(1, len(performance_elapsed_time))],
            index=[performance_elapsed_time[t] for t in range(1, len(performance_elapsed_time))], columns=['frequency'])
        data[cluster][trace]['aggregated_values']['execution_time'] = performance_elapsed_time[-1]
        data[cluster][trace]['aggregated_values']['upsampled_timestamps'] = data[cluster][trace]['rapl_sensors'].index
        # print("________________", folds)
        data[cluster][trace]['aggregated_values']['progress_frequency_median'] = pd.DataFrame({'median': np.nanmedian(
            data[cluster][trace]['aggregated_values']['performance_frequency']['frequency'].where(
                data[cluster][trace]['aggregated_values']['performance_frequency'].index <=
                data[cluster][trace]['aggregated_values']['upsampled_timestamps'][0], 0)), 'timestamp':
                                                                                                   data[cluster][trace][
                                                                                                       'aggregated_values'][
                                                                                                       'upsampled_timestamps'][
                                                                                                       0]}, index=[0])
        idx = 0
        data[cluster][trace]['aggregated_values']['pcap'] = pd.DataFrame(
            {'pcap': math.nan, 'timestamp': data[cluster][trace]['aggregated_values']['upsampled_timestamps'][0]},
            index=[0])
        for t in range(1, len(data[cluster][trace]['aggregated_values']['upsampled_timestamps'])):
            data[cluster][trace]['aggregated_values']['progress_frequency_median'] = \
            data[cluster][trace]['aggregated_values']['progress_frequency_median'].append({'median': np.nanmedian(
                data[cluster][trace]['aggregated_values']['performance_frequency']['frequency'].where((data[cluster][trace]['aggregated_values'][
                                                                                                           'performance_frequency'].index >=
                                                                                                       data[cluster][
                                                                                                           trace][
                                                                                                           'aggregated_values'][
                                                                                                           'upsampled_timestamps'][
                                                                                                           t - 1]) & (
                                                                                                                  data[
                                                                                                                      cluster][
                                                                                                                      trace][
                                                                                                                      'aggregated_values'][
                                                                                                                      'performance_frequency'].index <=
                                                                                                                  data[
                                                                                                                      cluster][
                                                                                                                      trace][
                                                                                                                      'aggregated_values'][
                                                                                                                      'upsampled_timestamps'][
                                                                                                                      t]))),
                                                                                           'timestamp':
                                                                                               data[cluster][trace][
                                                                                                   'aggregated_values'][
                                                                                                   'upsampled_timestamps'][
                                                                                                   t]},
                                                                                          ignore_index=True)

        pareto = {}

        data[cluster][trace]['aggregated_values']['energy'] = np.nansum([np.nansum([data[cluster][trace]['rapl_sensors']['value'+str(package_number)].iloc[i+1]*data[cluster][trace]['aggregated_values']['rapls_periods'].iloc[i][0] for i in range(0,len(data[cluster][trace]['aggregated_values']['rapls_periods'].index))]) for package_number in range(0,2)])

        pareto[cluster] = pd.DataFrame({'Execution Time':[data[cluster][trace]['aggregated_values']['progress_frequency_median']['median'].index[-1]]},index=[data[cluster][trace]['aggregated_values']['energy']/10**3])
        pareto[cluster].sort_index(inplace=True)
        if 'optimal' in folds:
            total_count += 1
            color = 'r'
            if flag_optimal:
                legend_name = None
            else:
                legend_name = 'optimal pcap'
                axes2.scatter(data[cluster][trace]['rapl_sensors']['value' + str(1)].index,
                              data[cluster][trace]['rapl_sensors']['value' + str(1)], c=color,
                              s=10, label=legend_name)
                flag_optimal = 1
            ET_list.append(pareto[cluster].values.tolist()[0][0])
            E_list.append(pareto[cluster].index.tolist()[0])
        elif 'minimum' in folds:
            color = 'b'
            if flag_minimal:
                legend_name = None
            else:
                legend_name = 'minimal pcap'
                axes2.scatter(data[cluster][trace]['rapl_sensors']['value' + str(1)].index,
                              data[cluster][trace]['rapl_sensors']['value' + str(1)], c=color,
                              s=10, label=legend_name)
                flag_minimal = 1
            ET_list_minimum.append(pareto[cluster].values.tolist()[0][0])
            E_list_minimum.append(pareto[cluster].index.tolist()[0])

        elif 'maximum' in folds:
            color = 'g'
            if flag_maximal:
                legend_name = None
            else:
                legend_name = 'maximal pcap'
                axes2.scatter(data[cluster][trace]['rapl_sensors']['value' + str(1)].index,
                              data[cluster][trace]['rapl_sensors']['value' + str(1)], c=color,
                              s=10, label=legend_name)
                flag_maximal = 1
            ET_list_maximum.append(pareto[cluster].values.tolist()[0][0])
            E_list_maximum.append(pareto[cluster].index.tolist()[0])


        execution_time = performance_elapsed_time[-1]
        print("Elapsed times are: ", execution_time)
        axes.scatter(pareto[cluster].index,execution_time, marker='.', c=color, s=100, label=legend_name)
        # axes.scatter(pareto[cluster].index,pareto[cluster]['Execution Time'], marker='.', c=color, s=100, label=legend_name)



        #axes2.scatter(data[cluster][trace]['rapl_sensors']['value'+str(1)],data[cluster][trace]['rapl_sensors']['value'+str(1)].index,c=np.array([colors(color_index)]), s=10, label=legend_name)

        plt.draw()
    except:
        print(folds)
        traceback.print_exc()
        pass
    color_index += 1



res_ET = stat.pstdev(ET_list)
print("Standard Deviation in Execution Time using optimal control is:",res_ET)
res_E = stat.pstdev(E_list)
print("Standar Deviation in Energy Consumption using optimal control is:", res_E)
mean_ET = stat.mean(ET_list)
print("Mean Execution time using optimal control is: ", mean_ET)
mean_E = stat.mean(E_list)
print("Mean Energy Consumption using optimal control is: ",mean_E)
axes.scatter(mean_E,mean_ET, marker='*', c='k', s=20, label=None)

res_ET_minimum = stat.pstdev(ET_list_minimum)
print("Standard Deviation in Execution Time for minimum control is:",res_ET_minimum)
res_E_minimum = stat.pstdev(E_list_minimum)
print("Standar Deviation in Energy Consumption for minimum control is:", res_E_minimum)
mean_ET_minimum = stat.mean(ET_list_minimum)
print("Mean of the minimal execution time is:", mean_ET_minimum)
mean_E_minimum = stat.mean(E_list_minimum)
print("Mean of the minimal energy consuption is: ", mean_E_minimum)
axes.scatter(mean_E_minimum ,mean_ET_minimum, marker='*', c='k', s=20, label=None)

res_ET_maximum = stat.pstdev(ET_list_maximum)
print("Standard Deviation in Execution Time for maximum control is:",res_ET_maximum)
res_E_maximum = stat.pstdev(E_list_maximum)
print("Standar Deviation in Energy Consumption for maximum control is:", res_E_maximum)
mean_ET_maximum = stat.mean(ET_list_maximum)
print("Mean of the minimal execution time is:", mean_ET_maximum)
mean_E_maximum = stat.mean(E_list_maximum)
print("Mean of the minimal energy consuption is: ", mean_E_maximum)
axes.scatter(mean_E_maximum,mean_ET_maximum, marker='*', c='k', s=20, label=None)



#label_params = axes.get_legend_handles_labels()
#axes_label.legend(*label_params, loc="center")

axes.grid(True)
axes.set_ylabel('Execution time [s]')
axes.set_xlabel('Energy consumption [kJ]')
leg = axes.legend()
axes.set_title(f'Plot of total energy consumed against execution time for {total_count} executions')

axes2.grid(True)
axes2.set_xlabel('Time [s]')
axes2.set_ylabel('Instantaneous power sensed by RAPL sensors')
leg2 = axes2.legend()
axes2.set_title(f'Plot of control action taken against time steps for {total_count} executions')

results_dir = './Results/'
if not os.path.exists(results_dir):
    os.makedirs(f'{results_dir}')
now = datetime.datetime.now()
fig.savefig(results_dir+'_'+str(now)+'_ENERGY_EXECUTION.pdf')
fig2.savefig(results_dir+'_'+str(now)+'_PCAP_TIME_STEPS.pdf')
#fig_label.savefig(results_dir+'_'+str(now)+'_LABEL_ONLY.pdf',bbox_inches='tight')

plt.show()

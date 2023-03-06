import argparse
import contextlib
import csv
import logging
import logging.config
import math
import pathlib
import statistics
import time
import uuid

import cerberus
import ruamel.yaml
import nrm.tooling as nrm

from gym import Env
from gym.spaces import Box
import numpy as np
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import math



exec_steps = 10000                              
ACTION_MIN = 40 
ACTION_MAX = 120
ACT_MID = ACTION_MIN + (ACTION_MAX - ACTION_MIN) / 2    
OBS_MAX = 60                                             
OBS_MIN = 0                                              
OBS_MID = OBS_MIN + (OBS_MAX - OBS_MIN) / 2   
EXEC_TIME = 100


RAPL_SENSOR_FREQ = 1
CPD_SENSORS_MAXTRY = 5

LOGGER_NAME = 'RL-runner'

LOGS_LEVEL = 'INFO'

LOGS_CONF = {
    'version': 1,
       'formatters': {
        'precise': {
            'format': '{created}\u0000{levelname}\u0000{process}\u0000{funcName}\u0000{message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': f'/tmp/{LOGGER_NAME}.log',
            'mode': 'w',
            'level': LOGS_LEVEL,
            'formatter': 'precise',
            'filters': [],
        },
    },
    'loggers': {
        LOGGER_NAME: {
            'level': LOGS_LEVEL,
            'handlers': [
                'file',
            ],
        },
    },
}


logging.config.dictConfig(LOGS_CONF)

logger = logging.getLogger(LOGGER_NAME)

DUMPED_MSG_TYPES = {
    'pubMeasurements',
    'pubProgress',
}

CSV_FIELDS = {
    'common': (
        'msg.timestamp',
        'msg.id',
        'msg.type',
    ),
    'pubMeasurements': (
        'sensor.timestamp',  # time
        'sensor.id',         # sensorID
        'sensor.value',      # sensorValue
    ),
    'pubProgress': (
        'sensor.cmd',    # cmdID
        'sensor.task',   # taskID
        'sensor.rank',   # rankID
        'sensor.pid',    # processID
        'sensor.tid',    # threadID
        'sensor.value',
    ),
}
assert DUMPED_MSG_TYPES.issubset(CSV_FIELDS)


def initialize_csvwriters(stack: contextlib.ExitStack):
    csvfiles = {
        msg_type: stack.enter_context(open(f'/tmp/dump_{msg_type}.csv', 'w'))
        for msg_type in DUMPED_MSG_TYPES
    }

    csvwriters = {
        msg_type: csv.DictWriter(csvfile, fieldnames=CSV_FIELDS['common']+CSV_FIELDS[msg_type])
        for msg_type, csvfile in csvfiles.items()
    }
    for csvwriter in csvwriters.values():
        csvwriter.writeheader()

    return csvwriters


def pubMeasurements_extractor(msg_id, payload):
    timestamp, measures = payload
    for data in measures:
        yield {
            'msg.timestamp': timestamp * 1e-6,  # convert µs in s
            'msg.id': msg_id,
            'msg.type': 'pubMeasurements',
            #
            'sensor.timestamp': data['time'] * 1e-6,  # convert µs in s
            'sensor.id': data['sensorID'],
            'sensor.value': data['sensorValue'],
        }


def pubProgress_extractor(msg_id, payload):
    timestamp, identification, value = payload
    yield {
        'msg.timestamp': timestamp * 1e-6,  # convert µs in s
        'msg.id': msg_id,
        'msg.type': 'pubProgress',
        #
        'sensor.cmd': identification['cmdID'],
        'sensor.task': identification['taskID'],
        'sensor.rank': identification['rankID'],
        'sensor.pid': identification['processID'],
        'sensor.tid': identification['threadID'],
        'sensor.value': value,
    }



def noop_extractor(*_):
    yield from ()


DUMPED_MSG_EXTRACTORS = {
    'pubMeasurements': pubMeasurements_extractor,
    'pubProgress': pubProgress_extractor,
}
assert DUMPED_MSG_TYPES.issubset(DUMPED_MSG_EXTRACTORS)


def dump_upstream_msg(csvwriters, msg):
    msg_id = uuid.uuid4()
    (msg_type, payload), = msg.items()  
    msg2rows = DUMPED_MSG_EXTRACTORS.get(msg_type, noop_extractor)
    csvwriter = csvwriters.get(msg_type)
    for row in msg2rows(msg_id, payload):
        csvwriter.writerow(row)



def abnormal_action(a):
    return a * ACTION_MIN + ACT_MID


def normal_obs(o):
    return (o - OBS_MIN) / (OBS_MAX - OBS_MIN)


def abnormal_obs(z):
    return z * (OBS_MAX - OBS_MIN) + OBS_MIN


def dump_upstream_msg(csvwriters, msg):
    msg_id = uuid.uuid4()
    (msg_type, payload), = msg.items()
    msg2rows = DUMPED_MSG_EXTRACTORS.get(msg_type, noop_extractor)
    csvwriter = csvwriters.get(msg_type)
    for row in msg2rows(msg_id, payload):
        csvwriter.writerow(row)



def pass_action(daemon,raple_actuators,csvwriters) :
    while not self.daemon.all_finished():
        msg = self.daemon.upstream_recv()
        dump_upstream_msg(csvwriters, msg)
        (msg_type, payload), = msg.items()
        if msg_type == 'pubProgress':
            self._update_progress(payload)
        elif msg_type == 'pubMeasurements':
            self._update_measure(payload)

def _update_progress(self, payload):
    timestamp, _, _ = payload
    timestamp *= 1e-6
    self.heartbeat_timestamps.append(timestamp)

@staticmethod
def _estimate_progress(heartbeat_timestamps):
    """Estimate the heartbeats' frequency given a list of heartbeats' timestamps."""
    return statistics.median(
        1 / (second - first)
        for first, second in zip(heartbeat_timestamps, heartbeat_timestamps[1:])
    )


def _update_measure(self, payload):
    timestamp, measures = payload
    timestamp *= 1e-6 
    for data in measures:
        if data['sensorID'].startswith('RaplKey'):
            window_duration = timestamp - self.rapl_window_timestamp
            progress_estimation = self._estimate_progress(self.heartbeat_timestamps)

            error = self.progress_setpoint - progress_estimation

            self.powercap_linear = \
                    window_duration * self._integral_gain * error + \
                    self._proportional_gain * (error - self.prev_error) + \
                    self.powercap_linear

            self.powercap_linear = max(
                min(
                    self.powercap_linear,
                    self._powercap_linear_max
                ),
     #            self._powercap_linear_min
                )

            powercap = self._delinearize(self.powercap_linear)

            self.prev_error = error

            self.rapl_window_timestamp = timestamp
            self.heartbeat_timestamps = self.heartbeat_timestamps[-1:]

            powercap = round(powercap)  
            enforce_powercap(self.daemon, self.rapl_actuators, powercap)

            break


def enforce_powercap(daemon, rapl_actuators, powercap):
    # for each RAPL actuator, create an action that sets the powercap to powercap
    # print("POWER CAP IS:", powercap)
    powercap = int(powercap[0])
    set_pcap_actions = [
        nrm.Action(actuator.actuatorID, powercap)
        for actuator in rapl_actuators
    ]
    logger.info(f'set_pcap={powercap}')
    #print("the power cap control generated given to the daemon is : ", set_pcap_actions)
    daemon.actuate(set_pcap_actions)


class Custom_env(Env):
    def __init__(self, exec_time, daemon, rapl_actuators, csvwriters, workload_cfg, c_0=1, c_1=1, c_2=0, c_3=0):
        self.action_space = Box(low=-1, high=1, shape=(1,))   
        self.observation_space = Box(low=np.float32(np.array([0.0])), high=np.float32(np.array([1.0])), shape=(1,))    
        self.state = np.random.rand(1, )                                                                               
        self.execution_time = exec_time                                                                                
        self.c_0 = c_0                                                                                                 
        self.c_1 = c_1                                                                                                 
        self.c_2 = c_2                                                                                                  
        self.c_3 = c_3                                                                                                  
        self.current_step = 0                                                                                           
        self.total_power = 0                                                                                            
        self.CSV = csvwriters
        self.action = None
        self.daemon = daemon
        self.rapl_actuators = rapl_actuators
        self.heartbeat_timestamps = []
        self.rapl_window_timestamp = time.time()  # start of current RAPL window
        self.workload_cfg = workload_cfg


    @staticmethod
    def _estimate_progress(heartbeat_timestamps):
        """Estimate the heartbeats' frequency given a list of heartbeats' timestamps."""
        #print("Check data: ",heartbeat_timestamps, heartbeat_timestamps[1:])
        return statistics.median(
            1 / (second - first)
            for first, second in zip(heartbeat_timestamps, heartbeat_timestamps[1:])
        )


    def _update_measure(self, timestamp):
        #print(timestamp)
        window_duration = timestamp - self.rapl_window_timestamp
        #print(window_duration)
        #print("____%_____",self.heartbeat_timestamps)
        progress_estimation = self._estimate_progress(self.heartbeat_timestamps)
        return progress_estimation
        
    def _update_progress(self, timestamp):
        timestamp *= 1e-6  # convert µs in s
        self.heartbeat_timestamps.append(timestamp)

    def step(self, action):
        
        #print("This is how hearbeats look like at this time", self.heartbeat_timestamps)

        actual_state = abnormal_obs(self.state) 
        power_cap = abnormal_action(action)
        logger.info(f"The current power cap given during the training is : {power_cap}")
        enforce_powercap(self.daemon, self.rapl_actuators, power_cap)
        measure_flag = 0
        while not measure_flag:
            msg = self.daemon.upstream_recv()
            #print("::::::::::::::::::::::::",msg)
            (msg_type,payload), = msg.items()
            dump_upstream_msg(self.CSV, msg)
            #print("_________________",msg_type)
            if msg_type == 'pubProgress':
                timestamp, _, _ = payload
                timestamp *= 1e-6
                #print("prog",timestamp)
                self._update_progress(timestamp)
            elif msg_type == 'pubMeasurements':
                timestamp, measures = payload
                timestamp *= 1e-6
                #print("measure",timestamp)
                for data in measures:
                    #print(data)
                    if data['sensorID'].startswith('RaplKey'):
                        new_state = self._update_measure(timestamp)
                        measure_flag = 1
                        break
        
        logger.info(f"The new state came as a result of the applied power cap is: {new_state}")
        normalized_new_state = normal_obs(new_state)                                                                   
        self.state = normalized_new_state                                                                           
        self.action = action[0]                                                                                        
        if new_state > 0:                                                                                           
          
            self.current_step += new_state                                                                       
            reward_0 = -self.c_0 * self.action                                                                         
            reward_1 = self.c_1 * self.state / self.action                                                             
            reward = reward_0 + reward_1                                                                               

        else:
            reward = -100                                                                                              
        print(reward)
        done = self.daemon.all_finished()
        if done:
            print("______________________________",done)

        info = {}                                                                                                      
        return self.state, reward, done, info                                                                          
    def reset(self):
        #time.sleep(1)
        #self.action = None
        val = random.choice(range(0, 1))
        self.state = np.float32(np.array([val]))
        self.execution_time = exec_steps
        self.current_step = 0
        self.total_power = 0
        #self.heartbeat_timestamps = []
        self.rapl_window_timestamp = time.time()
        self.daemon.run(**self.workload_cfg)
        return self.state



def update_sensors_list(daemon, known_sensors, *, maxtry=CPD_SENSORS_MAXTRY, sleep_duration=0.5):
    assert isinstance(known_sensors, list)

    new_sensors = []
    for _ in range(maxtry):
        new_sensors = [
            sensor
            for sensor in daemon.req_cpd().sensors()
            if sensor not in known_sensors
        ]
        if new_sensors:
            break  
        time.sleep(sleep_duration)

    known_sensors.extend(new_sensors)  
    return new_sensors



def collect_rapl_actuators(daemon):
    cpd = daemon.req_cpd()
    rapl_actuators = list(
        filter(
            lambda a: a.actuatorID.startswith('RaplKey'),
            cpd.actuators()
        )
    )
    logger.info(f'rapl_actuators={rapl_actuators}')
    return rapl_actuators






def launch_application(daemon_cfg, workload_cfg, *, sleep_duration=0.5):
    with nrm.nrmd(daemon_cfg) as daemon:
        rapl_actuators = collect_rapl_actuators(daemon)
        sensors = daemon.req_cpd().sensors()
        logger.info(f'daemon_sensors={sensors}')
        logger.info('launch workload')
        #daemon.run(**workload_cfg)
        #app_sensors = update_sensors_list(daemon, sensors, sleep_duration=sleep_duration)

        #if not app_sensors:
            #logger.critical('failed to get application-specific sensors')
            #raise RuntimeError('Unable to get application-specific sensors')
        #logger.info(f'app_sensors={app_sensors}')

        with contextlib.ExitStack() as stack:
            csvwriters = initialize_csvwriters(stack)
            env = Custom_env(EXEC_TIME, daemon, rapl_actuators, csvwriters, workload_cfg)
            model = PPO('MlpPolicy',env,verbose=1)
            model.learn(total_timesteps=150)
            



def run(cmd):
    #print(cmd)
    daemon_cfg = {
        'raplCfg': {
            'raplActions': [
                {'microwatts': 1_000_000 * powercap}
                for powercap in range(
                    round(ACTION_MIN),
                    round(ACTION_MAX) + 1
                )
            ],
        },
        'passiveSensorFrequency': {
            'hertz': RAPL_SENSOR_FREQ,
        },
        'verbose': 'Error',
    }
    workload_cfg = {
        'cmd': cmd[1],
        'args': cmd[2:],
        'sliceID': 'sliceID',
        'manifest': {
            'app': {
                'instrumentation': {
                    'ratelimit': {'hertz': 100_000_000},
                },
            },
        },
    }
    logger.info(f'daemon_cfg={daemon_cfg}')
    logger.info(f'workload_cfg={workload_cfg}')
    launch_application(daemon_cfg, workload_cfg)
    logger.info('successful execution')



def cli(args=None):

    parser = argparse.ArgumentParser()
    options, cmd = parser.parse_known_args(args)
    return options, cmd



if __name__ == "__main__":
    options,cmd = cli ()
    run(cmd)































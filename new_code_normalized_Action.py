from gym import Env                                                                                                     # Import gym for setting up the enviromnet
from gym.spaces import Box                                                                                              # Import Box for the continuous time action space initializations
import numpy as np                                                                                                      # import numpy
import random                                                                                                           # import random for variable intializations
import matplotlib.pyplot as plt                                                                                         # plotting tool
from stable_baselines3 import PPO                                                                                       # Import PPO from stable baselines3. For more information please refer to https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
import math                                                                                                             # mathematical functions

"""

Pre computed values for the mathematical model of clusters.
These are results taken from the regression of data points
depicted in the previous paper on classical control logic.
 
"""
a = {'gros': 0.83, 'dahu': 0.94, 'yeti': 0.89, 'CC_cluster':0.92}
b = {'gros': 7.07, 'dahu': 0.17, 'yeti': 2.91, 'CC_cluster':0.88}
alpha = {'gros': 0.047, 'dahu': 0.032, 'yeti': 0.023, 'CC_cluster':0.034}
beta = {'gros': 28.5, 'dahu': 34.8, 'yeti': 33.7, 'CC_cluster':29.5}
K_L = {'gros': 25.6, 'dahu': 42.4, 'yeti': 78.5, 'CC_cluster':40.1}
tau = 0.33


cluster = 'CC_cluster'                                                                                                        # The cluster of choice {'dahu','gros','yeti'}
"""
Sampling time T_S=1, is the time between two consecutive sampling of output (performance)
and the next pcap is chosen. 

"""
T_S = 1
exec_steps = 10000                                                                                                      # Total clock cycles needed for the execution of program.
ACTION_MIN = 40                                                                                                         # Minima of control space (power cap), Please do not change while using mathematical model for simulations.
ACTION_MAX = 120                                                                                                        # Maxima of control space
ACT_MID = ACTION_MIN + (ACTION_MAX - ACTION_MIN) / 2                                                                    # Midpoint of the control space to compute the normalized action space
OBS_MAX = 60                                                                                                            # Maxima of observation space (performance)
OBS_MIN = 0                                                                                                             # Minima of observation space
OBS_MID = OBS_MIN + (OBS_MAX - OBS_MIN) / 2                                                                             # Observation MID point.


def progress_funct(p_now, p_cap):                                                                                       # Function definition of the mathematical model of cluster performance and pcap relation.
    pcap_old_L = -np.exp(-alpha[cluster] * (a[cluster] * p_cap + b[cluster] - beta[cluster]))                           # Calculation of the PCAP for fitting it into the model.
    progress_value = K_L[cluster] * T_S / (T_S + tau) * pcap_old_L + tau / (T_S + tau) * (p_now - K_L[cluster]) + \
                     K_L[cluster]                                                                                       # Mathematical relation
    progress_NL = K_L[cluster] * (1 + pcap_old_L)                                                                       # Unwanted return variable
    return progress_value, progress_NL


"""

Convert a linear model action value to non linear value for testing it on the system.

"""
def NL_power(pcap_L):
    pcap = ((-math.log(-pcap_L) / alpha[cluster]) + beta[cluster] - b[cluster]) / a[cluster]                            #
    return pcap

"""

Convert a non linear model action value to linear value for sampling it out for the next computation.

"""

def L_power(pcap_NL):
    pcap_L = -np.exp(-alpha[cluster] * (a[cluster] * pcap_NL + b[cluster] - beta[cluster]))
    return pcap_L

"""

Convert a linear model performance value to non linear value for plotting.

"""


def NL_perf(perf_L):
    perf = perf_L + K_L[cluster]
    return perf

"""

Convert a non linear model performance value to linear value for plotting.

"""

def L_perf(perf_NL):
    perf_L = perf_NL - K_L[cluster]
    return perf_L

"""
Linear model for analysis only
"""


def progress_funct_L(p_now_L, p_cap_L):
    # p_now_L = L_perf(p_now_L)
    # p_cap_L = L_power(p_cap_L)
    progress_value = K_L[cluster] * T_S / (T_S + tau) * p_cap_L + tau / (T_S + tau) * p_now_L
    # progress_value = NL_perf(progress_value)
    return progress_value, None


"""
Normalizing and scaling functions 
"""
def abnormal_action(a):
    return a * ACTION_MIN + ACT_MID


def normal_obs(o):
    return (o - OBS_MIN) / (OBS_MAX - OBS_MIN)


def abnormal_obs(z):
    return z * (OBS_MAX - OBS_MIN) + OBS_MIN


"""

Define the environment class calling all the predefined functions inside them for generating the rewards and returns for an action.

"""

class Custom_env(Env):
    def __init__(self, exec_time, c_0=0, c_1=0, c_2=0, c_3=0):                                                          # Initialize with the total steps needed for execution and reward coefficients (default zero)
        self.action_space = Box(low=-1, high=1, shape=(1,))                                                             # Define continuous normalized action space between -1 and 1
        self.observation_space = Box(low=np.float32(np.array([0.0])), high=np.float32(np.array([1.0])), shape=(1,))     # Define continuous and normalized observation space (Just one variable consisting of the previous performance)
        self.state = np.random.rand(1, )                                                                                # Initialize the state or the first observation
        self.execution_time = exec_time                                                                                 # Initialize the execution time into a class variable
        self.c_0 = c_0                                                                                                  # coefficient for the reward_1
        self.c_1 = c_1                                                                                                  # coefficient for the reward_2
        self.c_2 = c_2                                                                                                  # coefficient for the reward_3
        self.c_3 = c_3                                                                                                  # coefficient for the reward_4
        self.current_step = 0                                                                                           # initialize the current step which should increment till exec_time
        self.total_power = 0                                                                                            # initialize total power for each episode
        self.action = None                                                                                              # initialize the action variable to store it for computing rewards

    def step(self, action):
        actual_state = abnormal_obs(self.state)                                                                         # using the first sampled observation from the observation space and scaling it
        actual_action = abnormal_action(action)                                                                         # scaling the sampled action to the non linear domain
        new_state, add_on = progress_funct(actual_state, actual_action)                                                 # use the sampled action and observation to generate the next action
        normalized_new_state = normal_obs(new_state)                                                                    # normalize the obtained observation
        self.state = normalized_new_state[0]                                                                            # make it the next state
        self.action = action[0]                                                                                         # store the action
        if new_state[0] > 0:                                                                                            # condition to check valid action and valid observation (may not be needed in practical scenario)
            # c = 1
            self.current_step += new_state[0]                                                                           # sum the progress to the current_step
            # reward_1 = c * (self.total_power/(120*175))
            # reward_2 = -self.c_1 * (self.action)
            # reward_1 = self.c_2 * (-self.execution_time + self.current_step)
            reward_0 = -self.c_0 * self.action                                                                          # compute the reward associated with minimizing the power consumed
            reward_1 = self.c_1 * self.state / self.action                                                              # compute the reward corresponding to the maximum performance given the action
            # reward_2 = self.c_2*(self.state)**2/self.action
            # reward_3 = self.c_3*(self.state)**3/self.action
            # reward_1 =  self.total_power
            # reward_2 = -self.action
            # reward_3 = -self.total_power

            reward = reward_0 + reward_1                                                                                # total reward

        else:
            reward = -100                                                                                               # if action resulted in something beyond the domain penalize the reward

        if self.current_step >= self.execution_time:                                                                    # stopping condition
            done = True
        else:
            done = False

        info = {}                                                                                                       # empty return variable generally used for plotting. I am using render function.
        return self.state, reward, done, info                                                                           # all return variables of the class

    def render(self, time):                                                                                             # plotting or render function
        axs[0].plot(time, abnormal_obs(self.state), 'g.')
        axs[0].set(xlabel='time', ylabel='performance')
        axs[1].plot(time, abnormal_action(self.action), 'r.')
        axs[1].set(xlabel='time', ylabel='power cap')

    def reset(self):                                                                                                    # reset function called at the beginning of all episodes even the first episode. Saves space for initialization.
        val = random.choice(range(0, 1))
        self.state = np.float32(np.array([val]))
        self.execution_time = exec_steps
        self.current_step = 0
        self.total_power = 0
        return self.state


def exec_man(c_0, c_3):                                                                                                 # main function
    env = Custom_env(exec_steps, c_0=c_0, c_1=c_3)                                                                      # define the environment with the c_0 and c_3 variables passed. We need only two of these to compute the rewards. We are not using the other functions.

    model = PPO("MlpPolicy", env, verbose=1)                                                                            # Define the PPO architecture (in-built)
    model.learn(total_timesteps=15000)                                                                                  # learning begins
    model.save("./models_"+str(cluster)+"/dynamics_" + str(c_0) + "___" + str(c_3))                                                      # saving the learnt models for a given c_0 and c_3

"""
uncomment the below lines for viewing the performance of all the learnt models and choose the best one among them.
"""

    # del model # remove to demonstrate saving and loading

    # model = PPO.load("dynamics")

    # count = 0
    # dones = False
    # obs = env.observation_space.sample()
    # energy = 0
    # print("______",obs)
    # while not dones:
    #     action, _states = model.predict(obs,deterministic=True)
    #     # print("____",action)
    #     energy += abnormal_action(action)
    #     # print(actual_action)
    #     obs, rewards, dones, info = env.step(action)
    #     obs = obs.reshape((1,))
    #     # print("_____",obs)
    #     count += 1
    # env.render(count)
    # plt.draw()
    # now = datetime.now()
    # plt.savefig("./figures_home_normal/result_"+str(c_1)+"____"+str(c_2)+"___"+str(now)+".png")
    # plt.close('all')
    # return(energy/1000)


if __name__ == "__main__":
    fig, axs = plt.subplots(2)
    fig.suptitle('power and performance against time')
    # C1_vals = [1]
    C0_vals = np.linspace(0, 20, 20)
    # C0_vals = [3]
    # C1_vals = np.linspace(0,20,20)
    # C2_vals = np.linspace(0,20,20)
    C3_vals = np.linspace(0, 10, 10)
    for i in C0_vals:
        # for j in C1_vals:
        #     for k in C2_vals:
        for l in C3_vals:
            exec_man(i, l)
    #plt.show()
    # plt.close('all')

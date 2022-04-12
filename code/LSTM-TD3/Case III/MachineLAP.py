"""
Machine class.
Serves as interface between agent functions and experimental implementation.

Modified from server class framework from:
    "Reinforcement Learning for Active Flow Control in Experiments" (https://arxiv.org/abs/2003.03419)
    https://github.com/LiuYangMage/RLFluidControl/blob/master/server/server.py
    By Dixia Fan, Liu Yang, Michael S Triantafyllou, and George Em Karniadakis

Updated to remove XML server aspects, for TF2, and for LSTM-TD3.

Modified by Peter Renn.
"""
import os
import pickle
import tensorflow as tf
import numpy as np
from agent_TD3_keras import TD3
from utils import KalmanFilter, NoneFilter
from datetime import datetime
import argparse

class MachineLAP():
    """
        'Machine' to interface between the agent functions and application.
    """
    def __init__(self, filter, *args, **kwargs):
        """
        args:
        	filter: filter to use for data(e.g. Kalman)
        """
        self.action_dim = 1
        # Dimension of state space
        self.state_dim = 10
        # Dimensionn of load
        self.load_dim = 1
        # Create agent_TD3 with the given state and action dimensions
        self.agent = TD3(self.state_dim, self.action_dim)
        # Stores unobservable state (i.e. the load)
        self.load_record = []
        # Stores unfiltered loads
        self.unfiltered_load_record = []
        # Stores recorded states (filtered)
        self.state_record = []
        # Stores recorded states (not filtered)
        self.unfiltered_state_record = []
        # Stores actions taken
        self.action_record = []
        # Stores timesteps
        self.time_record = []
        # Means for subtracting
        self.state_mean = np.zeros((1,self.state_dim))
        self.load_mean = np.zeros((1, self.load_dim))
        # Sets the filter to be used
        self.filter_name = filter
        if self.filter_name == "Kalman":
            self.filter = KalmanFilter()
            self.load_filter = KalmanFilter()
            self._stamp("Using Kalman Filter")
        elif self.filter_name == "None":
            self.filter = NoneFilter()
            self.load_filter = NoneFilter()
            self._stamp("Using No Filter")
        else:
            raise NotImplementedError
        # Sets the directory for saving
        self.save_model_dir = "save"
        self.save_data_dir = "save_data"
        self.save_eval_dir = "save_eval"

    def _stamp(self, string):
        """
        prints a time-stamped statement.

        args:
        	string: string to print a time-stamped statement
        """
        time = "UTC " + datetime.utcnow().isoformat(sep=" ", timespec="milliseconds") + " "
        print(time + string, flush = True)

    def _reward_func(self, this_state, this_load, next_state, next_load, last_load):
        """
        reward function used.

        args:
        	this_state: state for current timestep
            this_load: load for current timestep
            next_state: state for subsequent timestep
            next_load: load for subsequent timestep
            last_load: load for previous timestep
        """
        reward_raw = -10*(next_load)**2
        return reward_raw

    def _init(self, episode_count):
        """
        initialize agent

        args:
            episode_count: episode number from which to try to restore.
        """
        try:
            self._restore(episode_count)
            return True

        except Exception:
            self.agent.reset_agent()
            self._stamp("Initialized!")
            return False

    def _start_episode(self,state_mean, load_mean):
        """
        starts episode and resets buffers, records, and filters

        args:
            state_mean: mean state calculated/measured at beginning
            load_mean: mean load calculated/measured at beginning
        """
        self.state_mean = state_mean
        self.load_mean = load_mean
        self.load_record = []
        self.unfiltered_load_record = []
        self.state_record = []
        self.unfiltered_state_record = []
        self.action_record = []
        self.time_record = []
        self.agent.reset_episode()
        self.filter.reset()
        self.load_filter.reset()
        self._stamp("Episode Start!")
        return True

    def _start_eval(self,state_mean, load_mean):
        """
        starts evaluation period and resets buffers, records, and filters

        args:
            state_mean: mean state calculated/measured at beginning
            load_mean: mean load calculated/measured at beginning
        """
        self.state_mean = state_mean
        self.load_mean = load_mean
        self.load_record = []
        self.unfiltered_load_record = []
        self.state_record = []
        self.unfiltered_state_record = []
        self.action_record = []
        self.time_record = []
        self.filter.reset()
        self.load_filter.reset()
        self._stamp("Eval Start!")
        return True


    def _request_action(self, raw_data, raw_load, obs_buf, act_buf, stochastic):
        """
        requests action

        args:
            raw_data: raw (unfiltered) state data
            raw_load: raw (unfiltered) load data_
            obs_buf: observation buffer
            act_buf: action buffer
            stochastic: if true, add exploraiton noise (episode). if false, don't add noise (eval).
        """
        calibrated_load = raw_load - self.load_mean
        # Set the calibrated state to the raw value minus the mean
        calibrated_state = raw_data - self.state_mean
        # Filter the calibrated state
        filtered_state = self.filter.estimate(calibrated_state)
        # Filter the calibrated load
        filtered_load = self.load_filter.estimate(calibrated_load)
        # Get action
        action = self.agent.get_action(filtered_state, obs_buf, act_buf, stochastic = stochastic)
        self.unfiltered_load_record.append(calibrated_load)
        self.load_record.append(np.float64(filtered_load))
        self.unfiltered_state_record.append(calibrated_state)
        self.state_record.append(filtered_state)
        self.action_record.append(np.float64(action))
        raw_action = action[0,:].numpy().tolist()
        return raw_action

    def _request_stochastic_action(self, raw_data, raw_load, obs_buf, act_buf):
    """
    requests stochastic action for episodes.

    args:
        raw_data: raw (unfiltered) state data
        raw_load: raw (unfiltered) load data_
        obs_buf: observation buffer
        act_buf: action buffer
    """
        return self._request_action(raw_data, raw_load, obs_buf, act_buf, True)

    def _request_deterministic_action(self, raw_data, raw_load, obs_buf, act_buf):
    """
    requests deterministic action for eval.

    args:
        raw_data: raw (unfiltered) state data
        raw_load: raw (unfiltered) load data_
        obs_buf: observation buffer
        act_buf: action buffer
    """
        return self._request_action(raw_data, raw_load, obs_buf, act_buf, False)

    def _train(self, steps):
    """
    trains agent and saves episode.

    args:
        steps: number of steps to train for
    """
        self._stamp("Training..")
        # Makes directory for the data
        if not os.path.exists(self.save_data_dir):
            os.mkdir(self.save_data_dir)
        np.savez(self.save_data_dir + "/data_{}.npz".format(self.agent.episode_count),
                    state = np.array(self.state_record),
                    unfiltered_state = np.array(self.unfiltered_state_record),
                    action = np.array(self.action_record),
                    unfiltered_load  = np.array(self.unfiltered_load_record),
                    load = np.array(self.load_record),
                    time = np.array(self.time_record))
        # Get the length of the state records
        record_length = len(self.state_record)
        # Record the state record length
        self._stamp("Length of Record: " + str(record_length))
        # For each action and state, calculate the reward and store it
        for i in range(20, record_length-1):
            reward = self._reward_func(self.state_record[i],
                                       self.load_record[i],
                                       self.state_record[i+1],
                                       self.load_record[i+1],
                                       self.load_record[i-1])
            self.agent.replay_buffer.store(self.state_record[i],
                                        self.action_record[i],
                                        reward,
                                        self.state_record[i+1],
                                        0)

        # Stamp that training has started
        self._stamp("Training Start!")
        # Train for each step we saw
        for i in range(steps):
            self.agent.train_iter()
        self._stamp("Training End!")
        return True


    def _save(self):
    """
    saves agent.
    """
        if not os.path.exists(self.save_model_dir):
            os.mkdir(self.save_model_dir)
        self.agent.save_model()
        pickle_out = open(self.save_model_dir + "/{}.pickle".format(self.agent.episode_count),"wb")
        pickle.dump(self.agent.replay_buffer, pickle_out)
        pickle_out.close()
        self._stamp("Saved Episode {}!".format(self.agent.episode_count))
        return True


    def _restore(self,episode_count):
    """
    restores agent from previous run.

    args:
        episode_count: episode number
    """
        self.agent.saver.restore(self.agent.sess, self.save_model_dir + "/{}.ckpt".format(episode_count))
        pickle_in = open(self.save_model_dir + "/{}.pickle".format(episode_count),"rb")
        self.agent.replay_buffer = pickle.load(pickle_in)
        self.agent.episode_count = episode_count
        self._stamp("Restored from Episode {}!".format(episode_count))
        return True


    def _save_eval(self):
    """
    saves evaluation
    """
        if not os.path.exists(self.save_eval_dir):
            os.mkdir(self.save_eval_dir)

        np.savez(self.save_eval_dir + "/data_{}.npz".format(self.agent.episode_count),
                    state = np.array(self.state_record),
                    unfiltered_state = np.array(self.unfiltered_state_record),
                    action = np.array(self.action_record),
                    unfiltered_load  = np.array(self.unfiltered_load_record),
                    load = np.array(self.load_record),
                    time = np.array(self.time_record))
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="machine for wing")
    parser.add_argument("-fil", "--filter", choices=["None", "Kalman"], default="Kalman")
    args = parser.parse_args()

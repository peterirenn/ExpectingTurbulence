"""
Utilities

Kalman/None Filters odified from utils from:
    "Reinforcement Learning for Active Flow Control in Experiments" (https://arxiv.org/abs/2003.03419)
    https://github.com/LiuYangMage/RLFluidControl/blob/master/server/server.py
    By Dixia Fan, Liu Yang, Michael S Triantafyllou, and George Em Karniadakis

Replay modified buffer from:
    "Memory-based Deep Reinforcement Learning for POMDPs "(https://arxiv.org/pdf/2102.12344.pdf)
    https://github.com/LinghengMeng/LSTM-TD3
    by Lingheng Meng, Robert Gorbet, and Dana Kulic

Modified by Peter Renn.
"""

import numpy as np

# Basic Kalman filter
class KalmanFilter():

    def __init__(self, Q = 0.05, R = 10):
        self.Q_init = Q
        self.R_init = R
        self.reset()

    def reset(self):
        self.Q = self.Q_init
        self.R = self.R_init
        self.posteri = 0.0
        self.posteri_error = 1.0

    def estimate(self, measurement = None):
        if measurement is not None:
            priori = self.posteri
            priori_error = self.posteri_error + self.Q
            K = priori_error / (priori_error + self.R)
            self.posteri = priori + K * (measurement - priori)
            self.posteri_error = (1 - K) * priori_error
        return self.posteri


class NoneFilter():

    def __init__(self):
        pass

    def reset(self):
        self.posteri = 0.0

    def estimate(self, measurement = None):
        if measurement is not None:
            self.posteri = measurement
        return self.posteri

# Replays for soft actor critic agents
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.float32)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rews, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rews
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def sample_batch_with_history(self, batch_size=32, max_hist_len=10):
        idxs = np.random.randint(max_hist_len, self.size, size=batch_size)
        # History
        if max_hist_len == 0:
            hist_obs1 = np.zeros([batch_size, 1, self.obs_dim])
            hist_act = np.zeros([batch_size, 1, self.act_dim])
            hist_obs2 = np.zeros([batch_size, 1, self.obs_dim])
            hist_act2 = np.zeros([batch_size, 1, self.act_dim])
            hist_rews = np.zeros([batch_size, 1])
            hist_done = np.zeros([batch_size, 1])
            hist_len = np.zeros(batch_size)
        else:
            hist_obs1 = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_obs2 = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act2 = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_rews = np.zeros([batch_size, max_hist_len])
            hist_done = np.zeros([batch_size, max_hist_len])
            hist_len = max_hist_len * np.ones(batch_size)
            # Extract history experiences before sampled index
            for hist_i in range(max_hist_len):
                hist_obs1[:, -1 - hist_i, :] = self.obs1_buf[idxs - hist_i - 1, :]
                hist_act[:, -1 - hist_i, :] = self.acts_buf[idxs - hist_i - 1, :]
                hist_obs2[:, -1 - hist_i, :] = self.obs2_buf[idxs - hist_i - 1, :]
                hist_act2[:, -1 - hist_i, :] = self.acts_buf[idxs - hist_i, :]  # include a_t
                hist_rews[:, -1 - hist_i] = self.rews_buf[idxs - hist_i - 1, 0]
                hist_done[:, -1 - hist_i] = self.done_buf[idxs - hist_i - 1, 0]
            # If there is done in the backward experiences, only consider the experiences after the last done.
            for batch_i in range(batch_size):
                done_idxs_exclude_last_exp = np.where(hist_done[batch_i][:-1] == 1)  # Exclude last experience
                # If exist done
                if done_idxs_exclude_last_exp[0].size != 0:
                    largest_done_id = done_idxs_exclude_last_exp[0][-1]
                    hist_len[batch_i] = max_hist_len - (largest_done_id + 1)

                    # Only keep experiences after the last done
                    obs1_keep_part = np.copy(hist_obs1[batch_i, largest_done_id + 1:, :])
                    act_keep_part = np.copy(hist_act[batch_i, largest_done_id + 1:, :])
                    obs2_keep_part = np.copy(hist_obs2[batch_i, largest_done_id + 1:, :])
                    act2_keep_part = np.copy(hist_act2[batch_i, largest_done_id + 1:, :])
                    rews_keep_part = np.copy(hist_rews[batch_i, largest_done_id + 1:])
                    done_keep_part = np.copy(hist_done[batch_i, largest_done_id + 1:])

                    # Set to 0 to make sure all experiences are at the beginning
                    hist_obs1[batch_i] = np.zeros([max_hist_len, self.obs_dim])
                    hist_act[batch_i] = np.zeros([max_hist_len, self.act_dim])
                    hist_obs2[batch_i] = np.zeros([max_hist_len, self.obs_dim])
                    hist_act2[batch_i] = np.zeros([max_hist_len, self.act_dim])
                    hist_rews[batch_i] = np.zeros([max_hist_len])
                    hist_done[batch_i] = np.zeros([max_hist_len])

                    # Move kept experiences to the start of the segment
                    hist_obs1[batch_i, :max_hist_len - (largest_done_id + 1), :] = obs1_keep_part
                    hist_act[batch_i, :max_hist_len - (largest_done_id + 1), :] = act_keep_part
                    hist_obs2[batch_i, :max_hist_len - (largest_done_id + 1), :] = obs2_keep_part
                    hist_act2[batch_i, :max_hist_len - (largest_done_id + 1), :] = act2_keep_part
                    hist_rews[batch_i, :max_hist_len - (largest_done_id + 1)] = rews_keep_part
                    hist_done[batch_i, :max_hist_len - (largest_done_id + 1)] = done_keep_part
        #
        return dict(obs1=self.obs1_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.acts_buf[idxs],
                     rews=self.rews_buf[idxs],
                     done=self.done_buf[idxs],
                     hist_obs1=hist_obs1,
                     hist_act=hist_act,
                     hist_obs2=hist_obs2,
                     hist_act2=hist_act2,
                     hist_rews=hist_rews,
                     hist_done=hist_done,
                     hist_len=hist_len)

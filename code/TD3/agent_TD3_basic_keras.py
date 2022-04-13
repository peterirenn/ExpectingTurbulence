"""
Implementation of TD3 algorithm. 

Original found implemented in Pytorch at https://github.com/sfujim/TD3
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import ReplayBuffer, ProcessNoise

# Set seeed
seed = 41516
tf.random.set_seed(seed)
np.random.seed(seed)
class PolicyNet(keras.Model):
	""" policy network """
	def __init__(self, obs_dim, act_dim, nn_dim):
		"""
		args:
			obs_dim: dimension of the state of
			act_dim: dimension of the actionn
			nn_dim: list of nn dimensions
		"""
		super(PolicyNet, self).__init__()
		# Input
		self.input_2 = keras.layers.Dense(nn_dim[1], activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.33, distribution = 'uniform', seed=12365))
		# Dense layer
		self.dense_3 = keras.layers.Dense(nn_dim[2], activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.33, distribution = 'uniform', seed=17342))
		# Output
		self.output_1 = keras.layers.Dense(act_dim, activation='tanh',kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003, seed=10395))

	@tf.function
	def call(self, obs):
		x = self.input_2(obs)
		x = self.dense_3(x)
		x = self.output_1(x)
		return x


class QNets(keras.Model):
	""" twin critics """
	def __init__(self, obs_dim, act_dim, nn_dim):
		"""
		args:
			obs_dim: dimension of the state of
			act_dim: dimension of the actionn
			nn_dim: list of nn dimensions
		"""
		super(QNets, self).__init__()
		# QNet 1
		self.Q1_input_2 = keras.layers.Dense(nn_dim[1], activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.33, distribution = 'uniform', seed = 1391))
		self.Q1_dense_3 = keras.layers.Dense(nn_dim[2], activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.33, distribution = 'uniform', seed = 14390))
		self.Q1_output = keras.layers.Dense(1,kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003, seed = 10329))

		# QNet 2
		self.Q2_input_2 = keras.layers.Dense(nn_dim[1], activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.33, distribution = 'uniform', seed = 21312))
		self.Q2_dense_3 = keras.layers.Dense(nn_dim[2], activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.33, distribution = 'uniform', seed = 20329))
		self.Q2_output = keras.layers.Dense(1, activation = None,kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003, seed = 20330))

	@tf.function
	def call(self, obs, act):
		x0 = tf.concat([obs, act], 1)
		x1 = self.Q1_input_2(x0)
		x2 = self.Q2_input_2(x0)
		x1 = self.Q1_dense_3(x1)
		x1 = self.Q1_output(x1)
		x2 = self.Q2_dense_3(x2)
		x2 = self.Q2_output(x2)
		return x1, x2

	@tf.function
	def Q1(self, obs, act):
		x0 = tf.concat([obs, act], 1)
		x1 = self.Q1_input_2(x0)
		x1 = self.Q1_dense_3(x1)
		x1 = self.Q1_output(x1)
		return x1

	@tf.function
	def Q2(self, obs, act):
		x0 = tf.concat([obs, act], 1)
		x2_cf = self.Q2_input_2(x0)
		x2 = self.Q2_dense_3(x2)
		x2 = self.Q2_output(x2)
		return x2

class TD3(object):
	def __init__(
		self,
		obs_dim,
		act_dim,
	):

		"""
		args:
				obs_dim: dimension of state
				act_dim: dimension of action
		"""
		self.obs_dim = obs_dim
		self.act_dim = act_dim
		self.pi_nn_dim = [128, 128, 128, 128]
		self.qs_nn_dim = [128, 128, 128, 128]
		self.pi_lr = 1e-3
		self.qs_lr = 1e-3
		self.gamma = 0.99
		self.tau = 0.005
		self.bs = 50
		self.bfs = 50000
		self.d = 3
		self.pi_noise_std = 0.025
		self.pi_noise_clip = 0.5
		self.exploration_noise = 0.025

		self.pi = PolicyNet(self.obs_dim, self.act_dim, self.pi_nn_dim)
		self.qs = QNets(self.obs_dim, self.act_dim, self.qs_nn_dim)

		self.pi_targ = PolicyNet(self.obs_dim, self.act_dim, self.pi_nn_dim)
		self.qs_targ = QNets(self.obs_dim, self.act_dim, self.qs_nn_dim)

		T.assign(e) for (t,e) in zip(self.pi_targ.trainable, self.pi.trainable)
		T.assign(e) for (t,e) in zip(self.qs_targ.trainable, self.qs.trainable)

		self.pi_optimizer = keras.optimizers.Adam(lr=self.pi_lr, amsgrad = True)
		self.qs_optimzer = keras.optimizers.Adam(lr=self.qs_lr, amsgrad = True)

		self.qs_loss_fn = tf.keras.losses.Huber()

		self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, self.bfs)

		self.total_step_count = 0
		self.episode_count = 0

	def get_action(self, obs_data, stochastic = True):
		"""
			gets action from the policy net.
		"""
		action = self.pi.call(obs_data.reshape(1, -1)).numpy()
		if stochastic:
			explore_noise = tf.random.normal(action.shape, mean=0, stddev=self.exploration_noise)
			action = tf.clip_by_value(action + explore_noise, -1.0, 1.0)
		else:
			action = tf.clip_by_value(action, -1.0, 1.0)
		return action

	#@tf.function
	def train_iter(self):
		"""
			trains agent.
		"""
		if self.bs <= self.replay_buffer.size:
			# Sample batch
			this_batch = self.replay_buffer.sample_batch(self.bs)
			# Estimate next action
			next_action = self.pi_targ.call(this_batch["obs2"])
			# Ad policy noise for smoothing
			pi_noise = tf.random.normal(next_action.shape, mean=0, stddev=self.pi_noise_std)
			pi_noise = tf.clip_by_value(pi_noise, -self.pi_noise_clip, self.pi_noise_clip)
			noisy_next_action = tf.clip_by_value(next_action + pi_noise, -1.0, 1.0)
			# Find target values
			targ_q1, targ_q2 = self.qs_targ.call(this_batch["obs2"], noisy_next_action)
			targ_q = tf.minimum(targ_q1, targ_q2)
			# Update
			update_targs = tf.stop_gradient(this_batch["rews"] + (1 - this_batch["done"]) * self.gamma * targ_q)
			q_trainable = self.qs.trainable
			with tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(q_trainable)
				model_q1, model_q2 = self.qs(this_batch["obs1"], this_batch["act"])
				q_loss = (self.qs_loss_fn(update_targs, model_q1) + self.qs_loss_fn(update_targs, model_q2))

			q_grads = tape.gradient(q_loss, q_trainable)
			# Gradient clipping
			q_grads = [(tf.clip_by_value(grad, clip_value_min=-0.5, clip_value_max=0.5)) for grad in q_grads]
			self.qs_optimzer.apply_gradients(zip(q_grads, q_trainable))

			# Policy network update
			if self.total_step_count % self.d == 0:
				pi_trainable = self.pi.trainable
				with tf.GradientTape(watch_accessed_variables=False) as tape:
					tape.watch(pi_trainable)
					pi_loss = -tf.reduce_mean(self.qs.Q1(this_batch["obs1"],
							self.pi(this_batch["obs1"])))
				pi_grads = tape.gradient(pi_loss, pi_trainable)
				pi_grads = [(tf.clip_by_value(grad, clip_value_min=-0.5, clip_value_max=0.5)) for grad in pi_grads]

				self.pi_optimizer.apply_gradients(zip(pi_grads, pi_trainable))

				T.assign(T * (1 - self.tau) + M * self.tau) for (T,M) in zip(self.qs_targ.trainable, self.qs.trainable)
				T.assign(T * (1 - self.tau) + M * self.tau) for (T,M) in zip(self.pi_targ.trainable, self.pi.trainable)


			self.total_step_count += 1

	def reset_agent(self):
		"""
			resets agent.
		"""
		self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, self.bfs)
		self.total_step_count = 0
		self.episode_count = 0

		# Set targ to main variables
		T.assign(M) for (T,M) in zip(self.pi_targ.trainable, self.pi.trainable)
		T.assign(M) for (T,M) in zip(self.qs_targ.trainable, self.qs.trainable)

	def reset_episode(self):
		"""
			resets episode.
		"""
		self.step_count = 0
		self.episode_count += 1

	def save_model(self):
		"""
			aves the weights of everything
		"""
		self.pi.save_weights('./save_model/policy_/{}'.format(self.episode_count))
		self.pi_targ.save_weights('./save_model/policy_targ_/{}'.format(self.episode_count))
		self.qs.save_weights('./save_model/qnet_/{}'.format(self.episode_count))
		self.qs_targ.save_weights('./save_model/qnet_targ_/{}'.format(self.episode_count))

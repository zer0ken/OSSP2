import os
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512,
                name='critic', chkpt_dir='.\\tmp\\ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name + '_ddpg.h5')
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        input_ = tf.concat([state, action], axis=1)
        action_value = self.fc1(input_)
        action_value = self.fc2(action_value)
        q = self.q(action_value)
        return q


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2,
                name='critic', chkpt_dir='.\\tmp\\ddpg',
                action_min=None, action_max=None):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        if action_max is None and action_min is None:
            action_max = n_actions / 2
            action_min = -action_max
        
        self.action_min = action_min
        self.action_max = action_max
        
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name + '_ddpg.h5')
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(1, activation='sigmoid')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob)
        if self.action_min is not None and self.action_max is not None:
            mu = mu * (self.action_max - self.action_min) + self.action_min
        return mu
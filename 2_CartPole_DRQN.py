# Cartpole 
# State  -> x, x_dot, theta, theta_dot
# Action -> force (+1, -1)

import datetime
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
# Import modules
import tensorflow as tf

env = gym.make('CartPole-v0')
game_name = 'CartPole'
algorithm = 'DRQN'

# Parameter setting 
Num_action = 2
Gamma = 0.99
Learning_rate = 0.00025
Epsilon = 1
Final_epsilon = 0.01

Num_replay_memory = 200
Num_start_training = 5000
Num_training = 25000
Num_testing = 10000
Num_update = 250
Num_batch = 8
Num_episode_plot = 30

# DRQN Parameters
step_size = 4
lstm_size = 256
flatten_size = 4


# Initialize weights and bias
def weight_variable(shape):
    return tf.Variable(xavier_initializer(shape))


def bias_variable(shape):
    return tf.Variable(xavier_initializer(shape))


# Xavier Weights initializer
def xavier_initializer(shape):
    dim_sum = np.sum(shape)
    if len(shape) == 1:
        dim_sum += 1
    bound = np.sqrt(2.0 / dim_sum)
    return tf.random_uniform(shape, minval=-bound, maxval=bound)


# # Assigning network variables to target network variables
# def assign_network_to_target():
# 	update_wfc = tf.assign(w_fc_target, w_fc)
# 	update_bfc = tf.assign(b_fc_target, b_fc)

# 	sess.run(update_wfc)
# 	sess.run(update_bfc)

# 	cell_target = cell 

# Input 
x = tf.placeholder(tf.float32, shape=[None, 4])

w_fc = weight_variable([lstm_size, Num_action])
b_fc = bias_variable([Num_action])

rnn_batch_size = tf.placeholder(dtype=tf.int32)
rnn_step_size = tf.placeholder(dtype=tf.int32)

x_rnn = tf.reshape(x, [-1, rnn_step_size, flatten_size])

with tf.variable_scope('network'):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size, state_is_tuple=True)
    rnn_out, rnn_state = tf.nn.dynamic_rnn(inputs=x_rnn, cell=cell, dtype=tf.float32)

# Vectorization
rnn_out = rnn_out[:, -1, :]
rnn_out = tf.reshape(rnn_out, shape=[-1, lstm_size])

output = tf.matmul(rnn_out, w_fc) + b_fc

# # Target Network
# w_fc_target = weight_variable([lstm_size, Num_action])
# b_fc_target = bias_variable([Num_action])

# x_rnn_target = tf.reshape(x,[-1, rnn_step_size , flatten_size])

# with tf.variable_scope('target'):
# 	cell_target = tf.contrib.rnn.BasicLSTMCell(num_units = lstm_size, state_is_tuple = True)
# 	rnn_out_target, rnn_state_target = tf.nn.dynamic_rnn(inputs = x_rnn_target, cell = cell_target, dtype = tf.float32)

# # Vectorization
# rnn_out_target = rnn_out_target[:, -1, :]
# rnn_out_target = tf.reshape(rnn_out_target, shape = [-1 , lstm_size])

# output_target = tf.matmul(rnn_out_target, w_fc_target) + b_fc_target

# Loss function and Train 
action_target = tf.placeholder(tf.float32, shape=[None, Num_action])
y_prediction = tf.placeholder(tf.float32, shape=[None])

y_target = tf.reduce_sum(tf.multiply(output, action_target), reduction_indices=1)
Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
train_step = tf.train.AdamOptimizer(Learning_rate).minimize(Loss)

# Initialize variables
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)

# Initial parameters
Replay_memory = []
step = 1
score = 0
episode = 0

data_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(
    datetime.datetime.now().minute)

# DRQN variables
# Append episode data
episode_memory = []
observation_set = []

observation = env.reset()
action = env.action_space.sample()
observation, reward, terminal, info = env.step(action)

# Figure and figure data setting
plt.figure(1)
plot_x = []
plot_y = []

# Making replay memory
while True:
    # Rendering
    env.render()

    if step <= Num_start_training:
        state = 'Observing'
        action = np.zeros([Num_action])
        action[random.randint(0, Num_action - 1)] = 1.0
        action_step = np.argmax(action)

        observation_next, reward, terminal, info = env.step(action_step)
        reward -= 5 * abs(observation_next[0])

        if step % 100 == 0:
            print('step: ' + str(step) + ' / ' + 'state: ' + state)

    elif step <= Num_start_training + Num_training:
        # Training
        state = 'Training'

        # if random value(0 - 1) is smaller than Epsilon, action is random. Otherwise, action is the one which has the largest Q value
        if random.random() < Epsilon:
            action = np.zeros([Num_action])
            action[random.randint(0, Num_action - 1)] = 1.0
            action_step = np.argmax(action)

        else:
            Q_value = output.eval(feed_dict={x: observation_set, rnn_batch_size: 1, rnn_step_size: step_size})[0]
            action = np.zeros([Num_action])
            action[np.argmax(Q_value)] = 1
            action_step = np.argmax(action)

        observation_next, reward, terminal, info = env.step(action_step)
        reward -= 5 * abs(observation_next[0])

        # Select minibatch
        episode_batch = random.sample(Replay_memory, Num_batch)

        minibatch = []
        batch_end_index = []
        count_minibatch = 0

        for episode_ in episode_batch:
            episode_start = np.random.randint(0, len(episode_) + 1 - step_size)
            for step_ in range(step_size):
                minibatch.append(episode_[episode_start + step_])
                if step_ == step_size - 1:
                    batch_end_index.append(count_minibatch)

                count_minibatch += 1

        # Save the each batch data
        observation_batch = [batch[0] for batch in minibatch]
        action_batch = [batch[1] for batch in minibatch]
        reward_batch = [batch[2] for batch in minibatch]
        observation_next_batch = [batch[3] for batch in minibatch]
        terminal_batch = [batch[4] for batch in minibatch]

        # # Update target network according to the Num_update value
        # if step % Num_update == 0:
        # 	assign_network_to_target()

        # Get y_prediction
        y_batch = []
        action_in = []

        Q_batch = output.eval(
            feed_dict={x: observation_next_batch, rnn_batch_size: Num_batch, rnn_step_size: step_size})

        for count, i in enumerate(batch_end_index):
            action_in.append(action_batch[i])
            if terminal_batch[i] == True:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + Gamma * np.max(Q_batch[count]))

        train_step.run(
            feed_dict={action_target: action_in, y_prediction: y_batch, x: observation_batch, rnn_batch_size: Num_batch,
                       rnn_step_size: step_size})

        # Reduce epsilon at training mode
        if Epsilon > Final_epsilon:
            Epsilon -= 1.0 / Num_training

    elif step < Num_start_training + Num_training + Num_testing:
        # Testing
        state = 'Testing'
        Q_value = output.eval(feed_dict={x: observation_set, rnn_batch_size: 1, rnn_step_size: step_size})[0]

        action = np.zeros([Num_action])
        action[np.argmax(Q_value)] = 1
        action_step = np.argmax(action)

        observation_next, reward, terminal, info = env.step(action_step)

        Epsilon = 0

    else:
        # Test is finished
        print('Test is finished!!')
        plt.savefig('./Plot/' + data_time + '_' + algorithm + '_' + game_name + '.png')
        break

    # Save experience to the Replay memory
    episode_memory.append([observation, action, reward, observation_next, terminal])

    if len(Replay_memory) > Num_replay_memory:
        del Replay_memory[0]

    # Update parameters at every iteration
    step += 1
    score += reward

    observation = observation_next

    observation_set.append(observation)

    if len(observation_set) > step_size:
        del observation_set[0]

    # Plot average score
    if len(plot_x) % Num_episode_plot == 0 and len(plot_x) != 0 and state != 'Observing':
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Cartpole_DRQN')
        plt.grid(True)

        plt.plot(np.average(plot_x), np.average(plot_y), hold=True, marker='*', ms=5)
        plt.draw()
        plt.pause(0.000001)

        plot_x = []
        plot_y = []

    # Terminal
    if terminal == True:
        print('step: ' + str(step) + ' / ' + 'episode: ' + str(
            episode) + ' / ' + 'state: ' + state + ' / ' + 'epsilon: ' + str(Epsilon) + ' / ' + 'score: ' + str(score))

        if len(episode_memory) > step_size:
            Replay_memory.append(episode_memory)
        episode_memory = []

        # Plotting data
        plot_x.append(episode)
        plot_y.append(score)

        score = 0
        episode += 1
        observation = env.reset()

        observation_set = []
        for i in range(step_size):
            observation_set.append(observation)

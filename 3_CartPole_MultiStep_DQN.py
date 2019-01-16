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
algorithm = 'MultiStep'

# Parameter setting
Num_action = 2
Gamma = 0.99
Learning_rate = 0.00025
Epsilon = 1
Final_epsilon = 0.01

################### Multi-step ###################
# Parameter for Multi-step
n_step = 5
episode_step = 1
state_list = []
reward_list = []
##################################################

Num_replay_memory = 10000
Num_start_training = 10000
Num_training = 15000
Num_testing = 10000
Num_update = 150
Num_batch = 32
Num_episode_plot = 20

first_fc = [4, 512]
second_fc = [512, 128]
third_fc = [128, Num_action]

Is_render = False


# Initialize weights and bias
def weight_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


# Assigning network variables to target network variables
def assign_network_to_target():
    # Get trainable variables
    trainable_variables = tf.trainable_variables()
    # network variables
    trainable_variables_network = [var for var in trainable_variables if var.name.startswith('network')]

    # target variables
    trainable_variables_target = [var for var in trainable_variables if var.name.startswith('target')]

    for i in range(len(trainable_variables_network)):
        sess.run(tf.assign(trainable_variables_target[i], trainable_variables_network[i]))


# Input
x = tf.placeholder(tf.float32, shape=[None, 4])

# Densely connect layer variables
with tf.variable_scope('network'):
    w_fc1 = weight_variable('_w_fc1', first_fc)
    b_fc1 = bias_variable('_b_fc1', [first_fc[1]])

    w_fc2 = weight_variable('_w_fc2', second_fc)
    b_fc2 = bias_variable('_b_fc2', [second_fc[1]])

    w_fc3 = weight_variable('_w_fc3', third_fc)
    b_fc3 = bias_variable('_b_fc3', [third_fc[1]])

h_fc1 = tf.nn.relu(tf.matmul(x, w_fc1) + b_fc1)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

output = tf.matmul(h_fc2, w_fc3) + b_fc3

# Densely connect layer variables target
with tf.variable_scope('target'):
    w_fc1_target = weight_variable('_w_fc1', first_fc)
    b_fc1_target = bias_variable('_b_fc1', [first_fc[1]])

    w_fc2_target = weight_variable('_w_fc2', second_fc)
    b_fc2_target = bias_variable('_b_fc2', [second_fc[1]])

    w_fc3_target = weight_variable('_w_fc3', third_fc)
    b_fc3_target = bias_variable('_b_fc3', [third_fc[1]])

h_fc1_target = tf.nn.relu(tf.matmul(x, w_fc1_target) + b_fc1_target)
h_fc2_target = tf.nn.relu(tf.matmul(h_fc1_target, w_fc2_target) + b_fc2_target)

output_target = tf.matmul(h_fc2_target, w_fc3_target) + b_fc3_target

# Loss function and Train
action_loss = tf.placeholder(tf.float32, shape=[None, Num_action])
y_target = tf.placeholder(tf.float32, shape=[None])

y_prediction = tf.reduce_sum(tf.multiply(output, action_loss), reduction_indices=1)
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

plot_y_loss = []
plot_y_maxQ = []
loss_list = []
maxQ_list = []

data_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(
    datetime.datetime.now().minute)

state = env.reset()

# Figure and figure data setting
plot_x = []
plot_y = []

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

# Making replay memory
while True:
    if Is_render:
        # Rendering
        env.render()

    if step <= Num_start_training:
        progress = 'Exploring'
    elif step <= Num_start_training + Num_training:
        progress = 'Training'
    elif step < Num_start_training + Num_training + Num_testing:
        progress = 'Testing'
    else:
        # Test is finished
        print('Test is finished!!')
        plt.savefig('./Plot/' + data_time + '_' + algorithm + '_' + game_name + '.png')
        break

    # Select Action (Epsilon Greedy)
    if random.random() < Epsilon:
        action = np.zeros([Num_action])
        action[random.randint(0, Num_action - 1)] = 1.0
        action_step = np.argmax(action)
    else:
        Q_value = output.eval(feed_dict={x: [state]})[0]
        action = np.zeros([Num_action])
        action[np.argmax(Q_value)] = 1
        action_step = np.argmax(action)

    state_next, reward, terminal, info = env.step(action_step)

    if progress != 'Testing':
        # Training to stay at the center 
        reward -= 5 * abs(state_next[0])

    # Save experience to the Replay memory
    if len(Replay_memory) > Num_replay_memory:
        del Replay_memory[0]

    # Save experience to the Replay memory
    ###################################### Multi-step ######################################
    state_list.append(state)
    reward_list.append(reward)

    if episode_step > n_step:
        del state_list[0]
        del reward_list[0]

    if terminal:
        for i in range(n_step):
            reward_sum = 0
            for count, j in enumerate(range(i, n_step)):
                reward_sum += (Gamma ** count) * reward_list[j]
            Replay_memory.append([state_list[i], action, reward_sum, state_next, terminal])
    else:
        reward_sum = 0
        for i in range(len(reward_list)):
            reward_sum += (Gamma ** i) * reward_list[i]
        Replay_memory.append([state_list[0], action, reward_sum, state_next, terminal])

    ########################################################################################

    if progress == 'Training':
        minibatch = random.sample(Replay_memory, Num_batch)

        # Save the each batch data
        state_batch = [batch[0] for batch in minibatch]
        action_batch = [batch[1] for batch in minibatch]
        reward_batch = [batch[2] for batch in minibatch]
        state_next_batch = [batch[3] for batch in minibatch]
        terminal_batch = [batch[4] for batch in minibatch]

        y_batch = []

        # Update target network according to the Num_update value
        if step % Num_update == 0:
            assign_network_to_target()

        # Get y_prediction
        Q_batch = output_target.eval(feed_dict={x: state_next_batch})
        for i in range(len(minibatch)):
            if terminal_batch[i] == True:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + (Gamma ** n_step) * np.max(Q_batch[i]))

        loss, _ = sess.run([Loss, train_step], feed_dict={action_loss: action_batch,
                                                          y_target: y_batch,
                                                          x: state_batch})

        loss_list.append(loss)
        maxQ_list.append(np.max(Q_batch))

        # Reduce epsilon at training mode
        if Epsilon > Final_epsilon:
            Epsilon -= 1.0 / Num_training

    if progress == 'Testing':
        Epsilon = 0

    # Update parameters at every iteration
    step += 1
    episode_step += 1
    score += reward
    state = state_next

    # Plot average score
    if len(plot_x) % Num_episode_plot == 0 and len(plot_x) != 0 and progress != 'Exploring':
        ax1.plot(np.average(plot_x), np.average(plot_y_loss), '*')
        ax1.set_title('Mean Loss')
        ax1.set_ylabel('Mean Loss')
        ax1.hold(True)

        ax2.plot(np.average(plot_x), np.average(plot_y), '*')
        ax2.set_title('Mean score')
        ax2.set_ylabel('Mean score')
        ax2.hold(True)

        ax3.plot(np.average(plot_x), np.average(plot_y_maxQ), '*')
        ax3.set_title('Mean Max Q')
        ax3.set_ylabel('Mean Max Q')
        ax3.set_xlabel('Episode')
        ax3.hold(True)

        plt.draw()
        plt.pause(0.000001)

        plot_x = []
        plot_y = []
        plot_y_loss = []
        plot_y_maxQ = []

    # Terminal
    if terminal == True:
        print('step: ' + str(step) + ' / ' +
              'episode: ' + str(episode) + ' / ' +
              'progess: ' + progress + ' / ' +
              'epsilon: ' + str(Epsilon) + ' / ' +
              'score: ' + str(score))

        if progress != 'Exploring':
            # add data for plotting
            plot_x.append(episode)
            plot_y.append(score)
            plot_y_loss.append(np.mean(loss_list))
            plot_y_maxQ.append(np.mean(maxQ_list))

        score = 0
        loss_list = []
        maxQ_list = []
        episode += 1

        episode_step = 1
        state_list = []
        reward_list = []

        state = env.reset()

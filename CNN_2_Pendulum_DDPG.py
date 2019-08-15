import datetime
import random

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gym_plot

# Environment Setting
env = gym.make('Pendulum-v0')
game_name = 'Pendulum'
algorithm = 'DDPG'
pendulum_plot = gym_plot.Pendulum()
Num_states = env.observation_space.shape[0]
Num_action = env.action_space.shape[0]
action_max = 2

# Parameter setting
Gamma = 0.99
Learning_rate_actor = 0.0001
Learning_rate_critic = 0.001

Num_start_training = 5000
Num_training = 25000
Num_testing = 10000

Num_batch = 64
Num_replay_memory = 5000

Num_episode_plot = 10

# Network parameters
Num_colorChannel = 3
Num_stackFrame = 4

first_conv_actor = [8, 8, Num_colorChannel, 32]
second_conv_actor = [4, 4, 32, 32]
third_conv_actor = [3, 3, 32, 32]
first_fc_actor = [11 * 11 * 32, 200]
second_fc_actor = [200, 200]
third_fc_actor = [200, Num_action]

first_conv_critic = [8, 8, Num_colorChannel, 32]
second_conv_critic = [4, 4, 32, 32]
third_conv_critic = [3, 3, 32, 32]
first_fc_critic = [11 * 11 * 32, 400]
second_fc_critic = [400 + Num_action, 300]
third_fc_critic = [300, 1]

img_size = 84


# print('..........', Num_states, Num_action, '..........')
## Soft_update 调参数可以调整tau值
def Soft_update(Target_vars, Train_vars, tau=0.001):
    for v in range(len(Target_vars)):
        soft_target = sess.run(Train_vars[v]) * tau + sess.run(Target_vars[v]) * (1 - tau)
        Target_vars[v].load(soft_target, sess)


## Ornstein - Uhlenbeck noise
# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
# OU噪声
class OU_noise(object):
    def __init__(self, env_action, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.1, decay_period=Num_training):
        self.mu = mu  # 0.0
        self.theta = theta  # 0.15
        self.sigma = max_sigma  # 0.3
        self.max_sigma = max_sigma  # 0.3
        self.min_sigma = min_sigma  # 0.1
        self.decay_period = decay_period  # 25000
        self.num_actions = env_action.shape[0]  # 1
        self.action_low = env_action.low  # -2
        self.action_high = env_action.high  # 2
        self.reset()

    def reset(self):
        self.state = np.zeros(self.num_actions)

    # self.state = np.zeros(self.num_actions)
    # self.state = np.zeros(self.num_actions)
    def state_update(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.num_actions)  # np.random.randn()生成0,1的随机数
        self.state = x + dx

    def add_noise(self, action, training_step):
        self.state_update()
        state = self.state
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, training_step / self.decay_period)
        return np.clip(action + state, self.action_low, self.action_high)


def conv2d(x, w, stride):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')


def conv_weight_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())


# Initialize weights and bias
def weight_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def Actor(x, network_name):
    x_normalize = (x - (255.0 / 2)) / (255.0 / 2)
    # Actor Network
    with tf.variable_scope(network_name):
        w_conv1_actor = conv_weight_variable('_w_conv1', first_conv_actor)
        b_conv1_actor = bias_variable('_b_conv1', [first_conv_actor[3]])

        w_conv2_actor = conv_weight_variable('_w_conv2', second_conv_actor)
        b_conv2_actor = bias_variable('_b_conv2', [second_conv_actor[3]])

        w_conv3_actor = conv_weight_variable('_w_conv3', third_conv_actor)
        b_conv3_actor = bias_variable('_b_conv3', [third_conv_actor[3]])

        w_fc1_actor = weight_variable('_w_fc1', first_fc_actor)
        b_fc1_actor = bias_variable('_b_fc1', [first_fc_actor[1]])

        w_fc2_actor = weight_variable('_w_fc2', second_fc_actor)
        b_fc2_actor = bias_variable('_b_fc2', [second_fc_actor[1]])

        w_fc3_actor = weight_variable('_w_fc3', third_fc_actor)
        b_fc3_actor = bias_variable('_b_fc3', [third_fc_actor[1]])

    h_conv1_actor = tf.nn.relu(conv2d(x_normalize, w_conv1_actor, 4) + b_conv1_actor)
    h_conv2_actor = tf.nn.relu(conv2d(h_conv1_actor, w_conv2_actor, 2) + b_conv2_actor)
    h_conv3_actor = tf.nn.relu(conv2d(h_conv2_actor, w_conv3_actor, 2) + b_conv3_actor)
    h_pool3_flat = tf.reshape(h_conv3_actor, [-1, first_fc_actor[0]])

    h_fc1_actor = tf.nn.elu(tf.matmul(h_pool3_flat, w_fc1_actor) + b_fc1_actor)
    h_fc2_actor = tf.nn.elu(tf.matmul(h_fc1_actor, w_fc2_actor) + b_fc2_actor)

    output_actor = tf.nn.tanh(tf.matmul(h_fc2_actor, w_fc3_actor) + b_fc3_actor)
    # 个人理解，actor网络输出乘以动作的最大值，可能跟引入的噪声有关系，在引入噪声的过程中
    return action_max * output_actor


def reshape_input(state):
    state_out = cv2.resize(state, (img_size, img_size))
    state_out = np.uint8(state_out).reshape(1, img_size, img_size, Num_colorChannel)

    return state_out


def Critic(x, policy, network_name):
    x_normalize = (x - (255.0 / 2)) / (255.0 / 2)
    with tf.variable_scope(network_name):
        w_conv1_critic  = conv_weight_variable('_w_conv1', first_conv_critic)
        b_conv1_crititc = bias_variable('_b_conv1', [first_conv_critic[3]])

        w_conv2_critic = conv_weight_variable('_w_conv2', second_conv_critic)
        b_conv2_critic = bias_variable('_b_conv2', [second_conv_critic[3]])

        w_fc1_critic = weight_variable('_w_fc1', first_fc_critic)
        b_fc1_critic = bias_variable('_b_fc1', [first_fc_critic[1]])

        w_fc2_critic = weight_variable('_w_fc2', second_fc_critic)
        b_fc2_critic = bias_variable('_b_fc2', [second_fc_critic[1]])

        w_fc3_critic = weight_variable('_w_fc3', third_fc_critic)
        b_fc3_critic = bias_variable('_b_fc3', [third_fc_critic[1]])

    h_conv1_critic = tf.nn.relu(conv2d(x_normalize, w_conv1_critic, 4) + b_conv1_crititc)
    h_conv2_critic = tf.nn.relu(conv2d(h_conv1_critic, w_conv2_critic, 2) + b_conv2_critic)
    h_pool3_flat   = tf.reshape(h_conv2_critic, [-1, first_fc_critic[0]])
    # h_pool3_flat = tf.concat([h_pool3, policy], axis=1)

    # Critic Network
    h_fc1_critic = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1_critic) + b_fc1_critic)
    h_fc1_critic = tf.concat([h_fc1_critic, policy], axis=1)
    h_fc2_critic = tf.nn.relu(tf.matmul(h_fc1_critic, w_fc2_critic) + b_fc2_critic)

    output_critic = tf.matmul(h_fc2_critic, w_fc3_critic) + b_fc3_critic
    return output_critic


# Information from the network
# x = tf.placeholder(tf.float32, shape = [None, Num_states])
x = tf.placeholder(tf.float32, shape=[None, 84, 84, 3])

Policy = Actor(x, 'Actor_main')
Policy_target = Actor(x, 'Actor_target')
# tf.concat([T1, T2], 1) 后面参数为1，进行列合并
# 将下面的部分融入到critic网络中，由于不能够在卷积神经网络中直接加入动作，所以需要在全连接层加入
# Critic_inputs = tf.concat([Policy, x], 1)
# Critic_inputs_target = tf.concat([Policy, x], 1)
Q_Value = Critic(x, Policy, 'Critic_main')
Q_Value_target = Critic(x, Policy_target, 'Critic_target')

Actor_vars = tf.trainable_variables('Actor_main')
Actor_target_vars = tf.trainable_variables('Actor_target')

Critic_vars = tf.trainable_variables('Critic_main')
Critic_target_vars = tf.trainable_variables('Critic_target')

# Set Loss
target_critic = tf.placeholder(tf.float32, shape=[None, 1])
# tf.reduce_sum（）是求和公式，对所有变量进行求和。
# actor_loss = -tf.reduce_sum(Q_Value)
actor_loss = 1 / tf.reduce_sum(Q_Value)  # 最大化Q（自己修改的）
critic_loss = tf.losses.mean_squared_error(target_critic, Q_Value)  # 求解target_critic与Q的均方差

policy_optimizer = tf.train.AdamOptimizer(learning_rate=Learning_rate_actor)
critic_optimizer = tf.train.AdamOptimizer(learning_rate=Learning_rate_critic)

actor_train  = policy_optimizer.minimize(actor_loss, var_list=Actor_vars)
critic_train = critic_optimizer.minimize(critic_loss, var_list=Critic_vars)

# Init session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Initialization
noise = OU_noise(env.action_space)
# 为了能够随机探索，给其加上一个噪声处理过程，可以随机产生动作。噪声的最大值为2，最小值为-2.
state = env.reset()
state_img = pendulum_plot.get_state_img(state)
state_img = reshape_input(state_img)
noise.reset()
# Initial parameters
step = 0
step_train = 0
score = 0
episode = 0
data_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(
    datetime.datetime.now().minute)
replay_memory = []

# Figure and figure data setting
plot_loss = []
plot_Q = []
loss_list = []
maxQ_list = []

plot_x = []
plot_y = []

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

while True:
    # Define progress
    if step <= Num_start_training:
        progress = 'Exploring'
    # env.render()
    elif step <= Num_start_training + Num_training:
        progress = 'Training'
    # env.render()
    elif step < Num_start_training + Num_training + Num_testing:
        progress = 'Testing'
        env.render()
    else:
        # Test is finished
        print('Test is finished!!')
        plt.savefig('./Plot/' + data_time + '_' + algorithm + '_' + game_name + '.png')
        break

    # Choose action
    # state = state.reshape(-1, Num_states)
    # state_img = pendulum_plot.get_state_img(state)
    action = sess.run(Policy, feed_dict={x: state_img})

    # Add noise
    if progress != 'Testing':
        action = noise.add_noise(action, step_train)

    state_next, reward, terminal, _ = env.step(action)
    # state_next = state_next.reshape(-1, Num_states)# reshape（-1，n）表示的是行数未知，将其reshape为n列；将大小[3,1]变为[1,3]
    state_next = state_next.reshape(Num_states)
    state_next_img = pendulum_plot.get_state_img(state_next)
    state_next_img = reshape_input(state_next_img)
    # Experience replay 设置经验池
    if len(replay_memory) >= Num_replay_memory:
        del replay_memory[0]

    replay_memory.append([state_img, action, reward, state_next_img, terminal])

    if progress == 'Training':
        minibatch = random.sample(replay_memory, Num_batch)

        # Save the each batch data
        state_batch = [batch[0][0] for batch in minibatch]
        action_batch = [batch[1][0] for batch in minibatch]
        reward_batch = [batch[2][0] for batch in minibatch]
        state_next_batch = [batch[3][0] for batch in minibatch]
        terminal_batch = [batch[4] for batch in minibatch]

        # Update Critic
        y_batch = []
        Q_batch = sess.run(Q_Value_target, feed_dict={x: state_next_batch})

        for i in range(Num_batch):
            if terminal_batch[i]:
                y_batch.append([reward_batch[i]])
            else:
                y_batch.append([reward_batch[i] + Gamma * Q_batch[i][0]])

        _, loss_critic = sess.run([critic_train, critic_loss],
                                  feed_dict={target_critic: y_batch, x: state_batch, Policy: action_batch})

        # Update Actor
        _, loss_actor = sess.run([actor_train, actor_loss], feed_dict={x: state_batch})

        plot_loss.append(loss_critic)
        plot_Q.append(np.mean(Q_batch))

        ##Soft Update
        Soft_update(Actor_target_vars, Actor_vars)
        Soft_update(Critic_target_vars, Critic_vars)

        step_train += 1

    # Update parameters at every iteration
    step += 1
    score += reward[0]

    state_img = state_next_img

    # Plotting
    if len(plot_x) % Num_episode_plot == 0 and len(plot_x) != 0 and progress != 'Exploring':
        ax1.plot(np.average(plot_x), np.average(plot_y), '*')
        ax1.set_ylabel('Score')
        ax1.set_title('Average Score ' + algorithm)

        ax2.plot(np.average(plot_x), np.average(plot_loss), 'o')
        ax2.set_ylabel('Loss')
        ax2.set_title('Critic Loss ' + algorithm)

        ax3.plot(np.average(plot_x), np.average(plot_Q), 'd')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Q-value')
        ax3.set_title('Q_value ' + algorithm)

        plt.draw()
        plt.pause(0.000001)

        plot_x = []
        plot_y = []
        plot_loss = []
        plot_Q = []

    # Terminal
    if terminal:
        print('step: ' + str(step) + ' / ' + 'episode: ' + str(
            episode) + ' / ' + 'state: ' + progress + ' / ' + 'score: ' + str(score))

        if progress != 'Observing':
            # data for plotting
            plot_x.append(episode)
            plot_y.append(score)

        score = 0
        episode += 1

        state = env.reset()
        noise.reset()

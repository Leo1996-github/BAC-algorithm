import numpy as np
import tensorflow as tf
from env import env_class
import pandas as pd
import time
import copy
import math
import matplotlib.pyplot as plt
import random

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

env = env_class()
filename = r'instance.xlsx'
df1 = pd.read_excel(filename)
MAX_EPISODE = 100
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.0005    # learning rate for actor，Actor
LR_C = 0.001    # learning rate for critic，Critic
n0 = df1.loc[df1.iloc[:, 0].size - 1, 1:].values
L = df1.loc[0:df1.iloc[:, 0].size - 2, 0].values
Assemble_time = df1.loc[0:df1.iloc[:, 0].size - 2, 1:].values
C = 100
N = sum(n0)
m = len(n0)
N_F = int(m) + len(L)
N_A = int(m)
state_start = np.zeros(shape=(1, int(N)))
time_start = [0 for i in range(len(L))]

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.s_1 = tf.placeholder(tf.float32, [1, n_features], "state_1")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.a_1 = tf.placeholder(tf.int32, None, "act_1")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.Td_seq_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=30,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            l2 = tf.layers.dense(
                inputs=l1,
                units=20,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )
            l3 = tf.layers.dense(
                inputs=l2,
                units=10,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l3'
            )
            self.acts_prob = tf.layers.dense(
                inputs=l3,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )
        with tf.variable_scope('Actor_'):
            l1_1 = tf.layers.dense(
                inputs=self.s_1,
                units=30,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1_1'
            )
            l1_2 = tf.layers.dense(
                inputs=l1_1,
                units=20,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1_2'
            )
            l1_3 = tf.layers.dense(
                inputs=l1_2,
                units=10,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1_3'
            )
            self.acts_prob_1 = tf.layers.dense(
                inputs=l1_3,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob_1'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

        with tf.variable_scope('exp_v_1'):
            log_prob_ = tf.log(self.acts_prob_1[0, self.a_1])
            self.exp_v1 = tf.reduce_mean(log_prob_ * self.Td_seq_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train_1'):
            self.train_op1 = tf.train.AdamOptimizer(lr).minimize(-self.exp_v1)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def learn1(self, sa, deta):
        for i in range(len(sa)):
            s=sa[i][0][np.newaxis,:]
            a=sa[i][1]
            feed_dict = {self.s_1: s, self.a_1: a, self.Td_seq_error: deta}
            _, exp_v1 = self.sess.run([self.train_op1, self.exp_v1], feed_dict)

    def choose_action(self, s, MPS):
        s = s[np.newaxis, :]
        w = np.where(np.array(MPS) == 0)
        probs_=self.sess.run(self.acts_prob_1, {self.s_1: s})
        probs_.reshape(len(MPS), 1)
        loc_nan = np.argwhere(np.isnan(probs_))
        for i in loc_nan:
            probs_[i[0]] = 0
        if len(loc_nan) == len(MPS):
            probs_[:, :] = 1/len(MPS)
        probs_ = list(probs_)
        b1 = np.array(probs_).reshape(1, len(MPS))
        probs_=np.array([b_p for b_p in b1]).reshape(1,len(MPS))
        probs = self.sess.run(self.acts_prob, {self.s: s})
        probs.reshape(len(MPS), 1)
        loc_nan = np.argwhere(np.isnan(probs))
        for i in loc_nan:
            probs[i[0]] = 0
        if len(loc_nan)==len(MPS):
            probs[:,:] = 1/len(MPS)
        probs=list(probs)
        probs = np.array(probs).reshape(1,len(MPS))
        probs = np.array((np.dot(probs_, 0.8).ravel() + np.dot(probs,0.2)).ravel()).reshape(1, 4)
        probs=np.array([a_p / sum(probs.ravel()) for a_p in probs.ravel()]).reshape(1, 4)
        P = probs.ravel()
        if len(w[0]) == 0:
            P_new = P
            action = np.random.choice(np.arange(probs.shape[1]), p=P_new)
        else:

            for i in range(len(w[0])):
                P[w[0][i]] = 0
            P_0 = np.where(np.array(P) == 0)
            if len(P_0[0]) == len(MPS) :
                for i in range(len(P)):
                    if i not in w[0]:
                        P[i] = (len(P)-len(w[0]))/len(P)
            if sum(P)==0:
                P_new = [1 / len(P) for _ in range(len(P))]
            else:
                P_new = [a_p / sum(P) for a_p in P]
            action = np.random.choice(np.arange(probs.shape[1]), p=P_new)
        return action

class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=35,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            l2 = tf.layers.dense(
                inputs=l1,
                units=15,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )
            self.v = tf.layers.dense(
                inputs=l2,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error

sess = tf.Session()
actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor
sess.run(tf.global_variables_initializer())

def train():
    generation=0
    generation_sum=150
    while generation<generation_sum:
        for i_episode in range(MAX_EPISODE):
            acc_r=[0]
            td_error_acc = 0
            action_sequence=[]
            MPS=[random.randint(1,5) for _ in range(4)]
            Assemble_time_copy = copy.deepcopy(Assemble_time)
            time_start_copy=copy.deepcopy(time_start)
            state=np.append(np.array(MPS),np.array(time_start_copy))
            state_action_store=[[] for j in range(int(N))]
            for step in range(int(N)):
                a = actor.choose_action(state,MPS)
                s_, r_overload,time_start_,MPS_= env.step(a,L,Assemble_time_copy,C,time_start_copy,MPS)
                acc_r.append(r_overload+acc_r[-1])
                td_error = critic.learn(state,-r_overload , s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
                td_error_acc += abs(td_error.ravel())
                actor.learn(state, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]
                action_sequence.append(a)
                state_action_store[step]=[state,a]
                state = s_
                MPS=MPS_
                time_start_copy=time_start_
            print(acc_r[-1],action_sequence)
        generation += 1
        print("%d generation" % generation)

def Optimization():
    generation = 0
    generation_sum = 20
    ALL_EPI_reward = []
    min_reward = []
    TD_error_store = []
    while generation < generation_sum:
        EPI_state_action_store = [[] for i in range(MAX_EPISODE)]
        One_EPI_reward = []
        td_error_store = []
        for i_episode in range(MAX_EPISODE):
            acc_r = [0]
            td_error_acc = 0
            action_sequence = []
            MPS = n0.tolist()
            Assemble_time_copy = copy.deepcopy(Assemble_time)
            time_start_copy = copy.deepcopy(time_start)
            state=np.append(np.array(MPS),np.array(time_start_copy))
            state_action_store = [[] for j in range(int(N))]
            for step in range(int(N)):
                a = actor.choose_action(state, MPS)
                s_, r_overload, time_start_, MPS_ = env.step(a, L, Assemble_time_copy, C, time_start_copy, MPS)
                acc_r.append(r_overload + acc_r[-1])
                td_error = critic.learn(state, -r_overload, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
                td_error_acc += abs(td_error.ravel())
                actor.learn(state, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
                action_sequence.append(a)
                state_action_store[step] = [state, a]
                state = s_
                MPS = MPS_
                time_start_copy = time_start_
            td_error_store.extend(td_error_acc)
            One_EPI_reward.append(acc_r[-1])
            ALL_EPI_reward.append(acc_r[-1])
            print(acc_r[-1], action_sequence)
            EPI_state_action_store[i_episode] = state_action_store
        t_min = min(ALL_EPI_reward)
        TD_error_store.append(min(td_error_store))
        p = np.argsort(One_EPI_reward)
        for j in range(int(MAX_EPISODE / 3)):
            deta = -math.log((One_EPI_reward[p[j]] - t_min) / t_min + 0.00001)
            actor.learn1(EPI_state_action_store[p[j]], deta)
        generation += 1
        print("%d generation" % generation)
        min_reward.append(min(One_EPI_reward))

    f = open('BAC_MinReward.txt', 'w')
    for line in min_reward:
        f.write(str(line) + " ")
    f.close()
    return ALL_EPI_reward, t_min, TD_error_store,min_reward

train()
r,b,td_error_store,gen_best_Reward=Optimization()
print("MinOverload：",b)



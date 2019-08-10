"""
based on movan python reinforcement learning course
wrote some notes
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        #observation,action,reward
        self._build_net()#build network

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")#input
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")#labels
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")#
            #接收每个 state-action 所对应的 value (通过 reward 计算)
        #两层神经网络 ANN
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,#input
            units=10,#10 neural nodes
            activation=tf.nn.relu,  #  tanh activation function
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),#kernel init
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2 output lays
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,#softmax
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        #probability output
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            #first method：交叉熵：sum(label*log_act)
            # 最大化 总体 reward (log_p * R) 就是在最小化 -(log_p * R), 而 tf 的功能里只有最小化 loss
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:

            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
        #maxizmize result
        with tf.name_scope('train'):
            self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        #according to the probabilities of actions
        # add an vector became [observation,self.features]
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # select action w.r.t the actions prob
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())

        return action

    def store_transition(self, s, a, r):#memory round,not memory buffer
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):#this is the most important part which is different from others!
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards
        # feeding date
        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ] action_value feeding reward
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []#empty observation,action,value
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):#special procession
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):#reverse
            running_add = running_add * self.gamma + self.ep_rs[t]#最后面的是最新的，the oldest,the decay less
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)#result/summation std标准差
        return discounted_ep_rs




import numpy as np
import gym
from expert import tf_util

class GaussianSupervisor():

    def __init__(self, policy, cov):
        self.policy = policy
        self.cov = cov

    def sample_action(self, s):
        intended_action = self.policy.intended_action(s)
        sampled_action = np.random.multivariate_normal(intended_action, self.cov)
        return sampled_action


    def intended_action(self, s):
        return self.policy.intended_action(s)

class NetSupervisor():

    def __init__(self, policy_fn, sess):
        self.policy_fn = policy_fn
        self.sess = sess
        with self.sess.as_default():
            tf_util.initialize()

    def sample_action(self, s):
        with self.sess.as_default():
            intended_action = self.policy_fn(s[None,:])[0]
            return intended_action

    def intended_action(self, s):
        return self.sample_action(s)


class Supervisor():

    def __init__(self, est, sup=None):

        self.est = est

    def intended_action(self, s):
        #if self.one_class_error is not None:
        #    return self.one_class_error
        # return self.est.predict([s])[0]
        return np.clip(self.est.predict([s])[0], -5, 5)

    def sample_action(self, s):
        return self.intended_action(s)

class EpsSupervisor():
    """
        Discrete action space
    """
    def __init__(self, policy, action_space, eps = 0.0):
        self.policy = policy
        self.eps = eps
        self.action_space = action_space

    def sample_action(self, s):
        intended_action = self.policy.intended_action(s)
        if np.random.uniform(0, 1) > self.eps:
            return intended_action
        else:
            return np.random.choice(self.action_space)


    def intended_action(self, s):
        return self.policy.intended_action(s)

# if __name__ == '__main__':
#     env = gym.envs.make("CartPole-v0")
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n

#     T = 200

#     policy = pg.DiscretePG(state_dim, action_dim)
#     policy.load('supervisor.ckpt')

#     sup = EpsSupervisor(policy, eps = 0.3)



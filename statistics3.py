import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import IPython
import random
import os
import utils
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def linearize(sim, x, u, simulate_fn):
    F = utils.finite_diff1(u, simulate_fn)
    sim.set_x(x)
    f, _, _, _ = sim.step(u)
    return F, f


def compute_rec(agent, sim, x, a, simulate_fn, oc):
    F, f = linearize(sim, x, a, simulate_fn)

    n = sim.observation_space.shape[0]
    d = a.shape[0]

    Fu = F
    with tf.Graph().as_default():
        u_var = tf.Variable(a.reshape((len(a), 1)), dtype='float64')
        x_prime = tf.matmul(Fu, (a.reshape(d, 1) - u_var)) + f.reshape([n, 1])

        vectors = oc.support_vectors_
        alphas = oc.dual_coef_.flatten()


        s = 0.0
        for vec, alpha in zip(vectors, oc.dual_coef_.flatten()):
            s += alpha * np.exp(-oc.gamma * np.linalg.norm(vec - vectors[0])**2.0)
        print "rho: " + str(s)


        obj = 0.0
        for vec, alpha in zip(vectors, alphas):
            obj += alpha * tf.exp(-oc.gamma * tf.square(tf.norm(vec - x_prime)))
        obj = -obj

        opt = tf.train.GradientDescentOptimizer(.05)
        train = opt.minimize(obj)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # print "\tloss: " + str(sess.run(obj))
        # print "\n"
        for i in range(20):
            sess.run(train) 
            # print "loss: " + str(sess.run(obj))
            # print "value: " + str(sess.run(u_var))
            # print "\n"
        print "\tloss: " + str(sess.run(obj))
        # print "value: " + str(sess.run(u_var))

        value = sess.run(u_var)
    return value



def collect_traj(env, sim, agent, oc, T, visualize=False, early_stop=True, clipped = False):
    states = []
    intended_actions = []
    taken_actions = []
    scores = []
    mags = []

    s = env.reset()
    x = env.get_x()

    reward = 0.0
    count = 0

    for t in range(T):
        simulate_fn = utils.gen_simulate_step_control(sim, x)
        
        score = oc.decision_function([s])[0, 0]
        scores.append(score)
        print "score: " + str(score)
        
        a_intended = agent.intended_action(s)
        a = agent.sample_action(s)

        if score < .2:
            print "Using recovery..."
            a = compute_rec(agent, sim, x, a, simulate_fn, oc)
            
        next_s, r, done, _ = env.step(a)
        next_x = env.get_x()
        reward += r

        states.append(s)
        intended_actions.append(a_intended)
        taken_actions.append(a)

        s = next_s
        x = next_x

        if visualize:
            env.render()

    return states, intended_actions, taken_actions, reward


def collect_robust_traj_multiple(env, sim, agent, ocs, T, opt, visualize=False, early_stop=True, clipped = False):
    states = []
    intended_actions = []
    taken_actions = []
    scores = []
    mags = []

    s = env.reset()
    x = env.get_x()

    reward = 0.0
    count = 0

    for t in range(T):
        simulate_fn = utils.gen_simulate_step_control(sim, x)
        
        score = ocs[t].decision_function([s])[0, 0]
        scores.append(score)
        
        a_intended = agent.intended_action(s)
        a = agent.sample_action(s)

        if score < .2 and t < (T - 1):
            count += 1
            print "\tUsing recovery..."
            a = compute_rec(agent, sim, x, a, simulate_fn, ocs[t+1])
        mags.append(np.linalg.norm(a - a_intended))


        next_s, r, done, _ = env.step(a)
        next_x = env.get_x()
        reward += r

        states.append(s)
        intended_actions.append(a_intended)
        taken_actions.append(a)

        s = next_s
        x = next_x

        if visualize:
            env.render()

        if early_stop == True and done == True:
            print "Breaking"
            break
    freq = count / float(t + 1) 
    return states, intended_actions, taken_actions, reward, freq, scores, mags







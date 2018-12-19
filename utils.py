import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import IPython
from scipy.optimize import approx_fprime as fprime
from numba import jit

def stringify(lst):
    s = ""
    for el in lst:
        s += str(el) + '-'
    return s[:-1]

def clear():
    plt.ioff()
    plt.clf()
    plt.cla()
    plt.close()

@jit
def score(est, X, y):
    preds = est.predict(X)
    loss = np.sum(np.square(y - preds))
    res = np.sum(np.square(y - np.mean(y, axis=0)))
    return 1 - loss/res

def stringify(param):
    if isinstance(param, list):
        s = ""
        for el in param:
            s += str(el) + '-'
        return s[:-1]
    else:
        return str(param)



def trajs2data(trajs):
    for i in range(len(trajs)):
        traj = trajs[i]
        states, controls = zip(*traj)
        if i == 0:
            X = np.array(states)
            y = np.array(controls)
        else:
            X = np.vstack((X, np.array(states)))
            y = np.vstack((y, np.array(controls)))
    return X, y


def finite_diff1(x, f):
    """
        Jacobian
        First order finite difference approximation
        works with vector or scalar output functions f
    """
    delta = 1e-10

    fx = f(x)
    def g(x, i):
        return f(x)[i]

    if isinstance(fx, float):
        return fprime(x, f, delta)

    A = np.zeros((len(fx), len(x)))
    for i in range(len(fx)):
        A[i, :] = fprime(x, g, delta, i)

    return A

def finite_diff2(x, f):
    """
        Hessian
        Second order finite difference approximation
        works only with scalar output functions f
    """
    delta = 1e-10

    fprime = scipy.optimize.approx_fprime
    Q = np.zeros((len(x), len(x)))
    def g(x, i):
        res = fprime(x, f, delta)
        return res[i]
    for i in range(len(x)):
        Q[i, :] = fprime(x, g, delta, i)
    return Q

@jit
def mul(*args):
    res = args[0]
    for arg in args[1:]:
        res = np.dot(res, arg)
    return res

@jit
def breakup_C(C, n):
    c1 = C[:n, :n]
    c2 = C[:n, n:]
    c3 = C[n:, :n]
    c4 = C[n:, n:]
    return c1, c2, c3, c4

@jit
def breakup_c(c, n):
    return c[:n], c[n:]




def first_order_taylor(x, f):
    zero_order = f(x)
    slope = finite_diff1(x, f)
    offset = x

    def g(x_prime):
        zero_order = f(x)

    return zero_order, slope, offset

def gen_simulate_step_control(sim, x):
    """
        Special case when only the control is the input
        i.e. the state is held constant
    """
    def simulate(u):
        sim.set_x(x)
        s, _, _, _ = sim.step(u)
        x_next = sim.get_x()
        return s
    return simulate

def gen_simulate(sim):
    """
        Generates a function that simulates control u at specified state x. Use custom states
        The reason for generating a function to do this rather than actually doing it is that
        finite difference methods require only one input (the fixed point) rather than extra sim param
    """
    def simulate(xu):
        x, u = xu[:sim.state_dim], xu[sim.state_dim:]
        sim.set_x(x)
        sim.step(u)
        x_next = sim.get_x()
        cost = sim.get_cost(x, u)
        return x_next, cost
    return simulate

def gen_simulate_step(sim):
    """
        Generates a function that simulates the next state given control u at state x
        given simulator sim.
    """
    def simulate_step(xu):
        return gen_simulate(sim)(xu)[0]
    return simulate_step

def gen_simulate_cost(sim):
    """
        Generates a function that simulates the cost of taking action u at state x invariant of time
        given simulator sim
    """
    def simulate_cost(xu):
        return gen_simulate(sim)(xu)[1]
    return simulate_cost


def generate_dir(title, sub_dir, input_params):
    input_params = input_params.copy()
    if 'sess' in input_params:
        del input_params['sess']
    if 'env' in input_params:
        del input_params['env']
    if 'pi' in input_params:
        del input_params['pi']
    if 'sim' in input_params:
        del input_params['sim']
    if 'sup' in input_params:
        del input_params['sup']
    if 'filename' in input_params:
        del input_params['filename']
    if 'misc' in input_params:
        del input_params['misc']
    if 'plot_dir' in input_params:
        del input_params['plot_dir']
        del input_params['data_dir']
        del input_params['samples']
    if 'weights' in input_params:
        del input_params['weights']
        del input_params['id']
        del input_params['ufact']

    d = 'results/' + sub_dir + '/' + input_params['envname'] + '.pkl/' + title + '/arch' + stringify(input_params['arch']) + '/'
    keys = sorted(input_params.keys())
    for key in keys:
        param = stringify(input_params[key])
        d += key + param + '_'
    return d[:-1]


def generate_plot_dir(title, sub_dir, input_params):
    d = generate_dir(title, sub_dir, input_params)
    return d + '_plot/'

def generate_data_dir(title, sub_dir, input_params):
    d = generate_dir(title, sub_dir, input_params)
    return d + '_data/'


def extract_data(params, iters, title, sub_dir, ptype='reward'):
    means, sems = [], []
    for it in iters:
        params['iters'] = it

        path = generate_data_dir(title, sub_dir, params) + 'data.csv'
        data = pd.read_csv(path)
        arr = np.array(data[ptype])

        mean, sem = np.mean(arr), scipy.stats.sem(arr)
        means.append(mean)
        sems.append(sem)
    return np.array(means), np.array(sems)

def extract_time_data(params, iters, title, sub_dir):
    params['iters'] = iters
    path = generate_data_dir(title, sub_dir, params) + 'time_tests.txt'
    data = pd.read_csv(path)
    arr = np.array(data['time'])
    mean, sem = np.mean(arr), scipy.stats.sem(arr)
    return mean, sem

def filter_data(params, states, i_actions):
    T = params['t']
    if params['subsample'] == True:
        k = np.random.randint(0, T/50)
        states, i_actions = states[k::T/50], i_actions[k::T/50]
        return states, i_actions
    elif params['randsample'] == True:
        k = np.random.randint(0, len(states), 50)
        states, i_actions = np.array(states)[k], np.array(i_actions)[k]
        return list(states), list(i_actions)
    else:
        return states, i_actions



def plot(datas, labels, opt, title, colors=None):
    plt.style.use('ggplot')
    x = list(range(datas[0].shape[1]))
    for i, (data, label) in enumerate(zip(datas, labels)):
        mean_local = mean(data)
        ste_local = ste(data)
        if colors is not None:
            plt.plot(x, mean_local, label = label, color=colors[i])
            plt.fill_between(x, mean_local - ste_local, mean_local + ste_local, alpha=.3, color=colors[i])
        else:
            plt.plot(x, mean_local, label = label)
            plt.fill_between(x, mean_local - ste_local, mean_local + ste_local, alpha=.3)

    plt.legend()
    plt.title(title)
    # plt.show()
    # plt.savefig(opt.plot_dir + title + "_plot.png") 
    # clear()


def plot_show(datas, labels, title, colors=None):
    plt.style.use('ggplot')
    x = list(range(datas[0].shape[1]))
    for i, (data, label) in enumerate(zip(datas, labels)):
        mean_local = mean(data)
        ste_local = ste(data)
        if colors is not None:
            plt.plot(x, mean_local, label = label, color=colors[i])
            plt.fill_between(x, mean_local - ste_local, mean_local + ste_local, alpha=.3, color=colors[i])
        else:
            plt.plot(x, mean_local, label = label)
            plt.fill_between(x, mean_local - ste_local, mean_local + ste_local, alpha=.3)

    plt.legend()
    plt.title(title)
    plt.show()
    clear()




def ste(trial_rewards):
    if trial_rewards.shape[0] == 1:
        return np.zeros(trial_rewards.shape[1])
    return scipy.stats.sem(trial_rewards, axis=0)

def mean(trial_rewards):
    return np.mean(trial_rewards, axis=0)


def mean_sem(trial_data):
    s = ste(trial_data)
    m = mean(trial_data)
    return m, s


def eval_oc(oc, X):
    preds = oc.predict(X)
    err = len(preds[preds == -1]) / float(len(preds))
    return err


def fit_all(ocs, trajs):
    for t, oc in enumerate(ocs):
        X_train = []
        for traj in trajs:
            X_train.append(traj[t])
        oc.fit(X_train)

def eval_ocs(ocs, trajs):
    T = len(ocs)
    errs = np.zeros(T)
    trajs_array = np.array(trajs)
    try:
        for t in range(T):
            X = trajs_array[:, t, :]
            errs[t] = eval_oc(ocs[t], X)
    except IndexError:
        print "Index error, do something?"
        IPython.embed()
    return errs












import numpy as np
import scipy.stats
import IPython
import random
import utils

def eval_agent_statistics_cont(env, agent, sup, T, num_samples=1):
    """
        evaluate in the given environment along the agent's distribution
        for T timesteps on num_samples
    """
    losses = []
    for i in range(num_samples):
        # collect trajectory with states visited and actions taken by agent
        tmp_states, _, tmp_actions, _ = collect_traj(env, agent, T)
        sup_actions = np.array([sup.intended_action(s) for s in tmp_states])
        errors = (sup_actions - tmp_actions) ** 2.0
        # compute the mean error on that trajectory (may not be T samples since game ends early on failures)
        errors = np.sum(errors, axis=1)
        losses.append(np.mean(errors))

    # compute the mean and sem on averaged losses.
    return stats(losses)

def eval_sup_statistics_cont(env, agent, sup, T, num_samples = 1):
    """
        Evaluate on the supervisor's trajectory in the given env
        for T timesteps
    """

    losses = []
    for i in range(num_samples):
        # collect states made by the supervisor (actions are sampled so not collected)
        tmp_states, _, _, _ = collect_traj(env, sup, T)

        # get inteded actions from the agent and supervisor
        tmp_actions = np.array([ agent.intended_action(s) for s in tmp_states ])
        sup_actions = np.array([sup.intended_action(s) for s in tmp_states])
        errors = (sup_actions - tmp_actions) ** 2.0

        # compute the mean error on that traj
        errors = np.sum(errors, axis=1)
        losses.append(np.mean(errors))

    # generate statistics, same as above
    return stats(losses)


def eval_sim_err_statistics_cont(env, sup, T, num_samples = 1):
    losses = []
    for i in range(num_samples):
        tmp_states, int_actions, taken_actions, _ = collect_traj(env, sup, T)
        int_actions = np.array(int_actions)
        taken_actions = np.array(taken_actions)
        errors = (int_actions - taken_actions) ** 2.0
        errors = np.sum(errors, axis=1)
        losses.append(np.mean(errors))
    return stats(losses)

def eval_agent_statistics_disc(env, agent, sup, T, num_samples=1):
    """
        evaluate in the given environment along the agent's distribution
        for T timesteps on num_samples
    """
    losses = []
    for i in range(num_samples):
        tmp_states, _, tmp_actions, _ = collect_traj(env, agent, T)
        sup_actions = np.array([sup.intended_action(s) for s in tmp_states])
        errors = (-(sup_actions == tmp_actions)).astype(int)
        losses.append(np.mean(errors))

    return stats(losses)

def eval_sup_statistics_disc(env, agent, sup, T, num_samples = 1):
    """
        Evaluate on the supervisor's trajectory in the given env
        for T timesteps
    """
    losses = []
    for i in range(num_samples):
        tmp_states, _, _, _ = collect_traj(env, sup, T)
        tmp_actions = np.array([ agent.intended_action(s) for s in tmp_states ])
        sup_actions = np.array([sup.intended_action(s) for s in tmp_states])
        errors = (-(sup_actions == tmp_actions)).astype(int)
        losses.append(np.mean(errors))

    return stats(losses)

def eval_rewards(env, agent, T, num_samples=1):
    reward_samples = np.zeros(num_samples)
    for j in range(num_samples):
        _, _, _, reward = collect_traj(env, agent, T)
        reward_samples[j] = reward
    return np.mean(reward_samples)


def stats(losses):
    if len(losses) == 1: sem = 0.0
    else: sem = scipy.stats.sem(losses)

    d = {
        'mean': np.mean(losses),
        'sem': sem
    }
    return d

def ste(trial_rewards):
    if trial_rewards.shape[0] == 1:
        return np.zeros(trial_rewards.shape[1])
    return scipy.stats.sem(trial_rewards, axis=0)

def std(trial_rewards):
    if trial_rewards.shape[0] == 1:
        return np.zeros(trial_rewards.shape[1])
    return np.std(trial_rewards, axis=0)

def mean(trial_rewards):
    return np.mean(trial_rewards, axis=0)


def mean_sem(trial_data):
    s = ste(trial_data)
    m = mean(trial_data)
    return m, s

def mean_std(trial_data):
    s = std(trial_data)
    m = mean(trial_data)
    return m, s


ste = utils.ste
mean = utils.mean
mean_sem = utils.mean_sem


def evaluate_agent_cont(env, agent, sup, T, num_samples = 1):
    stats = eval_agent_statistics_cont(env, agent, sup, T, num_samples)
    return stats['mean']

def evaluate_sup_cont(env, agent, sup, T, num_samples = 1):
    stats = eval_sup_statistics_cont(env, agent, sup, T, num_samples)
    return stats['mean']

def evaluate_sim_err_cont(env, sup, T, num_samples = 1):
    stats = eval_sim_err_statistics_cont(env, sup, T, num_samples)
    return stats['mean']

def evaluate_agent_disc(env, agent, sup, T, num_samples = 1):
    stats = eval_agent_statistics_disc(env, agent, sup, T, num_samples)
    return stats['mean']

def evaluate_sup_disc(env, agent, sup, T, num_samples = 1):
    stats = eval_sup_statistics_disc(env, agent, sup, T, num_samples)
    return stats['mean']


def collect_traj(env, agent, T, visualize=False, early_stop=True, clipped = False):
    """
        agent must have methods: sample_action and intended_action
        Run trajectory on sampled actions
        record states, sampled actions, intended actions and reward
    """
    states = []
    intended_actions = []
    taken_actions = []

    s = env.reset()

    reward = 0.0

    for t in range(T):

        a_intended = agent.intended_action(s)
        a = agent.sample_action(s)
        if clipped:
            a = np.clip(a, -1, 1)
        next_s, r, done, _ = env.step(a)
        reward += r

        states.append(s)
        intended_actions.append(a_intended)
        taken_actions.append(a)

        s = next_s

        if visualize:
            env.render()


        if early_stop == True and done == True:
            print "Breaking"
            break
            
    return states, intended_actions, taken_actions, reward



def collect_robust_traj(env, agent, oc, T, visualize=False, early_stop=True):
    states = []
    intended_actions = []
    taken_actions = []
    scores = []

    s = env.reset()

    reward = 0.0
    count = 0
    def dec(u):
        x = env.get_x()
        s, _, _, _ = env.step(u)
        env.set_x(x)
        return oc.decision_function([s])[0, 0]

    for t in range(T):
        score = oc.decision_function([s])[0, 0]
        scores.append(score)

        a_intended = agent.intended_action(s)

        if score < .1:
            alpha = .1 
            count += 1
            a = a_intended
            for _ in range(20):
                a = a + alpha * utils.finite_diff1(a, dec)
            next_s, r, done, _ = env.step(a)
        else:
            a = agent.sample_action(s)
            next_s, r, done, _ = env.step(a)

        reward += r

        states.append(s)
        intended_actions.append(a_intended)
        taken_actions.append(a)

        s = next_s

        if visualize:
            env.render()


        if early_stop == True and done == True:
            print "Breaking"
            break

    freq = count / float(t + 1)
    return states, intended_actions, taken_actions, reward, freq, scores



def collect_score_traj(env, agent, oc, T, visualize=False, early_stop=True):
    states = []
    intended_actions = []
    taken_actions = []
    scores = []

    s = env.reset()

    reward = 0.0
    count = 0

    def dec(u):
        x = env.get_x()
        s, _, _, _ = env.step(u)
        env.set_x(x)
        return oc.decision_function([s])[0, 0]

    for t in range(T):
        score = oc.decision_function([s])[0, 0]
        scores.append(score)

        a_intended = agent.intended_action(s)

        a = agent.sample_action(s)
        next_s, r, done, _ = env.step(a)

        reward += r

        states.append(s)
        intended_actions.append(a_intended)
        taken_actions.append(a)

        s = next_s

        if visualize:
            env.render()

        if early_stop == True and done == True:
            print "Breaking"
            break

    freq = count / float(t + 1)
    return states, intended_actions, taken_actions, reward, freq, scores




def collect_robust_traj_multiple(env, agent, ocs, T, opt, visualize=False, early_stop=True, clipped = False):
    states = []
    intended_actions = []
    taken_actions = []
    scores = []
    mags = []

    s = env.reset()

    reward = 0.0
    count = 0


    for t in range(T):

        def dec(u):
            x = env.get_x()
            if clipped:
                u = np.clip(u, -1, 1)
            s, _, _, _ = env.step(u)
            env.set_x(x)
            return ocs[t+1].decision_function([s])[0, 0]

        score = ocs[t].decision_function([s])[0, 0]
        scores.append(score)

        a_intended = agent.intended_action(s)

        if score < .1 and t < (T - 1):
            alpha = .01 
            count += 1
            a = a_intended
            for _ in range(opt.grads):
                update_a = alpha * utils.finite_diff1(a, dec)
                a = a + update_a
            mags.append(np.linalg.norm(a - a_intended))
            if clipped:
                a = np.clip(a, -1, 1)
            next_s, r, done, _ = env.step(a)
        else:
            a = agent.sample_action(s)
            if clipped:
                a = np.clip(a, -1, 1)
            next_s, r, done, _ = env.step(a)
            mags.append(0.0)

        reward += r

        states.append(s)
        intended_actions.append(a_intended)
        taken_actions.append(a)

        s = next_s

        if visualize:
            env.render()


        if early_stop == True and done == True:
            print "Breaking"
            break

    freq = count / float(t + 1)
    return states, intended_actions, taken_actions, reward, freq, scores, mags



def collect_score_traj_multiple(env, agent, ocs, T, visualize=False, early_stop=True, clipped=False):
    states = []
    intended_actions = []
    taken_actions = []
    scores = []

    s = env.reset()

    reward = 0.0
    count = 0


    for t in range(T):

        # def dec(u):
        #     x = env.get_x()
        #     if clipped:
        #         u = np.clip(u, -1, 1)
        #     s, _, _, _ = env.step(u)
        #     env.set_x(x)
        #     return ocs[t+1].decision_function([s])[0, 0]


        score = ocs[t].decision_function([s])[0, 0]
        scores.append(score)

        a_intended = agent.intended_action(s)

        a = agent.sample_action(s)
        if clipped:
            a = np.clip(a, -1, 1)

        next_s, r, done, _ = env.step(a)

        reward += r

        states.append(s)
        intended_actions.append(a_intended)
        taken_actions.append(a)

        s = next_s

        if visualize:
            env.render()

        if early_stop == True and done == True:
            print "Breaking"
            break

    freq = count / float(t + 1)
    return states, intended_actions, taken_actions, reward, freq, scores


def collect_traj_alt(env, agent, T, visualize=False, early_stop=True, clipped = False):
    """
        alternate version that uses internal states
    """
    states = []
    intended_actions = []
    taken_actions = []

    s = env.reset()
    x = env.get_x()

    reward = 0.0

    for t in range(T):

        a_intended = agent.intended_action(s)
        a = agent.sample_action(s)
        if clipped:
            a = np.clip(a, -1, 1)
        next_s, r, done, _ = env.step(a)
        next_x = env.get_x()
        reward += r

        states.append(x)
        intended_actions.append(a_intended)
        taken_actions.append(a)

        s = next_s
        x = next_x

        if visualize:
            env.render()


        if early_stop == True and done == True:
            print "Breaking"
            break
            
    return states, intended_actions, taken_actions, reward





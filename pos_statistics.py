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

def mean(trial_rewards):
    return np.mean(trial_rewards, axis=0)


def mean_sem(trial_data):
    s = ste(trial_data)
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
    d = env.action_space.shape[0]

    for t in range(T):

        score = ocs[t].decision_function([s])[0, 0]
        scores.append(score)

        a_intended = agent.intended_action(s)

        for i in range(10):
            if score < .025 and t < (T - 1):
                recovery_control = np.zeros(d)
                count += 1
                print "Activating recovery control: " + str(t)
                rand_control = np.random.normal(0, .01, (d))
                s_next, _, _, _ = env.step(rand_control)
                score_next = ocs[t].decision_function([s_next])[0, 0]
                if score_next < score:
                    a = - 2 * rand_control
                    s_next, _, _, _ = env.step(a)
                score_updated = ocs[t].decision_function([s_next])[0, 0]
                print "\t\toriginal score: " +  str(score)
                print "\t\tnext score: "     +  str(score_next)
                print "\t\tupdated score: "  +  str(score_updated)
                s = s_next
                mags.append(np.linalg.norm(a))

        a = agent.sample_action(s)
        mags.append(0.0)

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

        score = ocs[t].decision_function([s])[0, 0]
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

def collect_traj_rejection(env, agent, T, visualize=False, early_stop=True, reset = True):
    """
        agent must have methods: sample_action and intended_action
        Run trajectory on sampled actions
        record states, sampled actions, intended actions and reward
    """
    states = []
    intended_actions = []
    taken_actions = []
    info = {}

    s = env.reset()

    failed = False

    reward = 0.0

    for t in range(T):

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

        failed = env.violation()


        if early_stop == True and done == True:
            print "Breaking"
            break
            

    return states, intended_actions, taken_actions, reward, failed



def collect_score_traj_multiple_rejection(env, agent, ocs, T, visualize=False, early_stop=True):
    states = []
    intended_actions = []
    taken_actions = []
    scores = []
    # s = env.reset()
    s = env._get_obs()

    reward = 0.0
    count = 0
    freq = 0

    reject = False
    failed = False


    info = {}
    info['first_out'] = -1
    info['first_violation'] = -1
    info['first_complete'] = -1
    info['count_failures'] = 0
    info['count_fail_in_support'] = 0
    info['initial_state'] = env.get_pos_vel()



    if ocs[0].predict([s])[0] == -1:
        print "Initial state predicted out of distribution"
        reject = True

    else:
        for t in range(T):

            score = ocs[t].decision_function([s])[0, 0]
            scores.append(score)

            failed = env.violation()
            if failed:
                info['count_failures'] += 1
            if failed and score > 0:
                info['count_fail_in_support'] += 1

            if score < 0.0 and info['first_out'] == -1:
                info['first_out'] = t
            if failed and info['first_violation'] == -1:
                info['first_violation'] = t
            completed = env.completed()
            if completed and info['first_complete'] == -1:
                info['first_complete'] = t


            a_intended = agent.intended_action(s)
            a = agent.sample_action(s)
            print "norm: " + str(np.linalg.norm(a))

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
        if info['count_failures'] > 0:
            info['freq_fail_in_support'] = info['count_fail_in_support'] / float(info['count_failures'])
        else:
            info['freq_fail_in_support'] = 0.0
    return states, intended_actions, taken_actions, reward, freq, scores, info, failed, reject






def collect_robust_traj_multiple_rejection_adaptive(env, agent, ocs, T, opt, KLs, visualize=False, early_stop=True):
    states = []
    intended_actions = []
    taken_actions = []
    scores = []
    mags = []
    # s = env.reset()
    s = env._get_obs()
    reward = 0.0
    count = 0
    freq = 0
    d = env.action_space.shape[0]

    reject = False
    failed = False

    info = {}
    info['first_out'] = -1
    info['first_violation'] = -1
    info['rec_failed'] = -1
    info['first_complete'] = -1
    info['true_scores'] = []
    info['triggered'] = 0
    info['improved'] = 0

    info['count_failures'] = 0
    info['count_fail_in_support'] = 0
    info['initial_state'] = env.get_pos_vel()

    info['obj_positions'] = []
    info['fail_positions'] = []
    info['sup_viol_positions'] = []

    if ocs[0].predict([s])[0] == -1:
        print "Initial state predicted out of distribution"
        reject = True

    else:
        for t in range(T):
            triggered = False
            score = ocs[t].decision_function([s])[0, 0]
            info['true_scores'].append(score)
            score_last = score

            failed = env.violation()
            if failed:
                info['count_failures'] += 1
            if failed and score > 0:
                info['count_fail_in_support'] += 1

            if score < 0.0 and info['first_out'] == -1:
                info['first_out'] = t

            if failed and info['first_violation'] == -1:
                info['first_violation'] = t

            completed = env.completed()
            if completed and info['first_complete'] == -1:
                info['first_complete'] = t

            info['obj_positions'].append(env.get_body_com('object'))
            info['sup_viol_positions'].append(score < 0)
            info['fail_positions'].append(failed)

            a_intended = agent.intended_action(s)

            # print "\toriginal score: " +  str(score) + ", at " + str(t)
 
            j = 0
            print "\t\t" + str(t) + ", cutoff: " + str(KLs[t] * np.linalg.norm(a_intended))
            print "distances: " + str(ocs[t].mapping([s]))
            print "\t\tscore last: " + str(score_last)
            while score_last < (KLs[t] * np.linalg.norm(a_intended)) and not score_last < 0:
                triggered = True
                true_delta_u = np.zeros(d)
                delta_u = np.zeros(d)
                delta = max(score_last, 0.0) / 500.0 / KLs[t]
                for k in range(d):
                    delta_u[k] = delta / float(d)
                    delta_x, _, _, _ = env.step(delta_u)
                    delta_score = ocs[t].decision_function([delta_x])[0, 0]
                    if delta_score >= score_last:
                        true_delta_u[k] = 1.0
                    else:
                        true_delta_u[k] = -1.0

                    delta_u[k] = 0.0

                eta = np.abs(delta_score) / 150.0 / KLs[t]
                if delta <= 0:
                    u_r = np.zeros(true_delta_u.shape)
                else:
                    u_r = eta * true_delta_u / np.linalg.norm(true_delta_u)

                s, _, _, _ = env.step(u_r)
                env.render()
                score_last = ocs[t].decision_function([s])[0, 0]

                a_intended = agent.intended_action(s)
                print "time step: " + str(t) + ", j: " + str(j)
                print "learner norm: " + str(np.linalg.norm(a_intended))
                print "cutoff: " + str(KLs[t] * np.linalg.norm(a_intended))
                print "score last: " + str(score_last)
                print "Completed: " + str(env.completed())
                print "Failure: " + str(env.violation())
                print "\n"
                j += 1

            if triggered:
                info['triggered'] += 1
            if triggered and score_last > score:
                info['improved'] += 1

            print "\t\tscore updated: " + str(score_last)
            print ""
            if triggered:
                print "Recovery activated"

            score_updated = score_last
            if score > 0 and score_updated < 0 and info['rec_failed'] == -1:
                info['rec_failed'] = t


            a = agent.sample_action(s)
            mags.append(0.0)
            scores.append(score_updated)
            if score_updated > 0 or True:
                next_s, r, done, _ = env.step(a)
            else:
                next_s, r, done, _ = env.step(np.zeros(a.shape))
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
        if info['triggered'] > 0:    
            info['frac_improved'] = float(info['improved']) / info['triggered']
        else:
            info['frac_improved'] = 1.0
        info['frac_triggered'] = float(info['triggered']) / (t + 1)

    return states, intended_actions, taken_actions, reward, freq, scores, mags, info, failed, reject






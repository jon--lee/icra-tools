import numpy as np
import scipy.stats
import IPython
import random
import utils


max_rec = 500



def collect_traj(env, agent, ocs, T, visualize=False, early_stop = True, init_state = None):
    states = []
    intended_actions = []
    taken_actions = []

    s = env.reset()
    if init_state:
        env.set_pos_vel(*init_state)
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

    info['initial_state'] = env.get_pos_vel()

    info['completed'] = False
    info['failed'] = False
    info['failed_in_support'] = False

    if ocs[0].predict([s])[0] == -1:
        print "Initial state predicted out of distribution"
        reject = True
    else:
        for t in range(T):
            failed = env.violation()
            # print "\t\tscore: " + str(ocs[t].decision_function([s])[0,0])
            if failed and info['first_violation'] == -1:
                info['failed'] = True
                if info['first_out'] == -1:
                    info['failed_in_support'] = True
                info['first_violation'] = t

            if failed:
                break

            completed = env.completed()
            if completed and info['first_complete'] == -1:
                info['completed'] = True
                info['first_complete'] = t
                break

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



    return states, intended_actions, taken_actions, reward, info, failed, reject


"""
    Recovery roll outs will always monitor the value of the decision function.
    If the task is completed or a failure occurs, the trajectory will always
    be interrupted.

    If the robot leaves the support then the trajecotry will also stop.
"""
def collect_rec(recovery_loop, env, sim, agent, ocs, T, opt, KLs, visualize=False, early_stop = True, init_state = None, max_rec = max_rec):
    states = []
    intended_actions = []
    taken_actions = []
    scores = []
    mags = []

    s = env.reset()
    if init_state:
        env.set_pos_vel(*init_state)
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
    info['triggered'] = False

    info['initial_state'] = env.get_pos_vel()

    info['completed'] = False
    info['failed'] = False
    info['failed_in_support'] = False

    info['rec_scores'] = []
    info['rec_cutoffs'] = []

    if ocs[0].predict([s])[0] == -1:
        print "Initial state predicted out of distribution"
        reject = True

    else:
        for t in range(T):
            triggered = False
            score = ocs[t].decision_function([s])[0, 0]
            score_last = score

            if score < 0.0 and info['first_out'] == -1:
                info['first_out'] = t

            failed = env.violation()
            if failed and info['first_violation'] == -1:
                info['failed'] = True
                if info['first_out'] == -1:
                    info['failed_in_support'] = True
                info['first_violation'] = t

            if failed or score < 0.0:
                break

            completed = env.completed()
            if completed and info['first_complete'] == -1:
                info['completed'] = True
                info['first_complete'] = t
                break

            s, score_last, rec_info = recovery_loop(s, t, score_last, env, sim, agent, ocs, T, opt, KLs, visualize=visualize, max_rec = max_rec)
            a_intended = rec_info['a_intended']

            # if not info['triggered']:
            #     info['rec_scores'] = rec_info['rec_scores'] 
            #     info['rec_cutoffs'] = rec_info['rec_cutoffs']           

            if rec_info['triggered']:
                info['triggered'] = True
                info['rec_scores'] += [rec_info['rec_scores']]
                info['rec_cutoffs'] += [rec_info['rec_cutoffs']]

            if rec_info['reached_max']:
                print "\t\tNot able to recover, stopping"
                break

            score_updated = rec_info['score_last']
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
    return states, intended_actions, taken_actions, reward, freq, scores, mags, info, failed, reject





def random_sample_loop(s, t, score, env, sim, agent, ocs, T, opt, KLs, visualize=False, max_rec = 500):
    rec_info = {'triggered': False, 'reached_max': False}
    j = 0
    a_intended = agent.intended_action(s)
    score_last = score
    d = env.action_space.shape[0]
    rec_scores = np.zeros(max_rec + 1)
    rec_cutoffs = np.zeros(max_rec + 1)

    while j < max_rec and score_last < (KLs[t] * np.linalg.norm(a_intended)) and score_last > 0:
        # print "\t\tcutoff:     " + str(KLs[t] * np.linalg.norm(a_intended))
        # print "\t\tscore_last: " + str(score_last)
        # print "\t\tfailed:     " + str(env.violation()) + "\n"

        rec_info['triggered'] = True
        rec_scores[j] = score_last
        rec_cutoffs[j] = np.linalg.norm(a_intended) * KLs[t]

        delta_u = np.random.normal(0, .01, env.action_space.shape)
        delta = max(score_last, 0.0) / 500.0 / KLs[t]
        delta_u = delta * delta_u / np.linalg.norm(delta_u)

        delta_x, _, _, _ = env.step(delta_u)
        delta_score = ocs[t].decision_function([delta_x])[0, 0]
        if delta_score < score_last:
            delta_u = - delta_u
        if delta_score == score_last:
            delta_u = np.zeros(d)

        eta = np.abs(delta_score) / 50.0 / KLs[t]
        if delta <= 0:
            u_r = np.zeros(delta_u.shape)
        else:
            u_r = eta * delta_u / delta

        s, _, _, _ = env.step(u_r)
        if visualize:
            env.render()
        score_last = ocs[t].decision_function([s])[0, 0]

        a_intended = agent.intended_action(s)
        j += 1

    # print "\t\tcutoff:     " + str(KLs[t] * np.linalg.norm(a_intended))
    # print "\t\tscore_last: " + str(score_last)
    # print "\t\tfailed:     " + str(env.violation()) + "\n"


    rec_scores[j] = score_last
    rec_scores[j+1:] = score_last
    rec_cutoffs[j:] = KLs[t] * np.linalg.norm(a_intended)

    if rec_info['triggered']:
        print "\t\tRecovery was activated, stopped at t: " + str(t) + ", j: " + str(j)


    if j == max_rec:
        rec_info['reached_max'] = True

    rec_info['score_last'] = score_last
    rec_info['rec_scores'] = rec_scores
    rec_info['rec_cutoffs'] = rec_cutoffs
    rec_info['a_intended'] = a_intended
    return s, score_last, rec_info




def random_sample_loop_momentum(s, t, score, env, sim, agent, ocs, T, opt, KLs, visualize=False, max_rec = 500):
    rec_info = {'triggered': False, 'reached_max': False}
    j = 0
    a_intended = agent.intended_action(s)
    score_last = score
    d = env.action_space.shape[0]
    rec_scores = np.zeros(max_rec + 1)
    rec_cutoffs = np.zeros(max_rec + 1)

    last_u_r = np.zeros(d)
    alpha = .75

    while j < max_rec and score_last < (KLs[t] * np.linalg.norm(a_intended)) and score_last > 0:
        # print "\t\tcutoff:     " + str(KLs[t] * np.linalg.norm(a_intended))
        # print "\t\tscore_last: " + str(score_last)
        # print "\t\tfailed:     " + str(env.violation()) + "\n"

        rec_info['triggered'] = True
        rec_scores[j] = score_last
        rec_cutoffs[j] = np.linalg.norm(a_intended) * KLs[t]

        delta_u = np.random.normal(0, .01, env.action_space.shape)
        delta = max(score_last, 0.0) / 500.0 / KLs[t]
        delta_u = delta * delta_u / np.linalg.norm(delta_u)

        delta_x, _, _, _ = env.step(delta_u)
        delta_score = ocs[t].decision_function([delta_x])[0, 0]
        if delta_score < score_last:
            delta_u = - delta_u
        if delta_score == score_last:
            delta_u = np.zeros(d)

        eta = np.abs(delta_score) / 50.0 / KLs[t]
        if delta <= 0:
            u_r = np.zeros(delta_u.shape)
        else:
            u_r = eta * delta_u / delta
        u_r = (1.0 - alpha) * u_r + alpha * last_u_r

        s, _, _, _ = env.step(u_r)
        if visualize:
            env.render()
        score_last = ocs[t].decision_function([s])[0, 0]

        a_intended = agent.intended_action(s)

        last_u_r = u_r
        j += 1

    # print "\t\tcutoff:     " + str(KLs[t] * np.linalg.norm(a_intended))
    # print "\t\tscore_last: " + str(score_last)
    # print "\t\tfailed:     " + str(env.violation()) + "\n"


    rec_scores[j] = score_last
    rec_scores[j+1:] = score_last
    rec_cutoffs[j:] = KLs[t] * np.linalg.norm(a_intended)

    if rec_info['triggered']:
        print "\t\tRecovery was activated, stopped at t: " + str(t) + ", j: " + str(j)


    if j == max_rec:
        rec_info['reached_max'] = True

    rec_info['score_last'] = score_last
    rec_info['rec_scores'] = rec_scores
    rec_info['rec_cutoffs'] = rec_cutoffs
    rec_info['a_intended'] = a_intended
    return s, score_last, rec_info





def approx_grad_loop(s, t, score, env, sim, agent, ocs, T, opt, KLs, visualize=False, max_rec = 500):
    rec_info = {'triggered': False, 'reached_max': False}
    d = env.action_space.shape[0]
    j = 0
    a_intended = agent.intended_action(s)
    score_last = score
    d = env.action_space.shape[0]

    rec_scores = np.zeros(max_rec + 1)
    rec_cutoffs = np.zeros(max_rec + 1)

    
    while j < max_rec and score_last < (KLs[t] * np.linalg.norm(a_intended)) and score_last > 0:
        rec_info['triggered'] = True
        rec_scores[j] = score_last
        rec_cutoffs[j] = np.linalg.norm(a_intended) * KLs[t]

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

        eta = np.abs(delta_score) / 50.0 / KLs[t]
        if delta <= 0:
            u_r = np.zeros(true_delta_u.shape)
        else:
            u_r = eta * true_delta_u / np.linalg.norm(true_delta_u)

        s, _, _, _ = env.step(u_r)
        if visualize:
            env.render()
        score_last = ocs[t].decision_function([s])[0, 0]

        a_intended = agent.intended_action(s)
        j += 1


    rec_scores[j] = score_last
    rec_scores[j+1:] = score_last
    rec_cutoffs[j:] = KLs[t] * np.linalg.norm(a_intended)

    if rec_info['triggered']:
        print "\t\tRecovery was activated, stopped at t: " + str(t) + ", j: " + str(j)


    if j == max_rec:
        rec_info['reached_max'] = True

    rec_info['score_last'] = score_last
    rec_info['rec_scores'] = rec_scores
    rec_info['rec_cutoffs'] = rec_cutoffs
    rec_info['a_intended'] = a_intended
    return s, score_last, rec_info






def finite_diff_loop(s, t, score, env, sim, agent, ocs, T, opt, KLs, visualize=False, max_rec = 500):
    rec_info = {'triggered': False, 'reached_max': False}
    j = 0
    d = env.action_space.shape[0]
    a_intended = agent.intended_action(s)
    score_last = score
    d = env.action_space.shape[0]

    rec_scores = np.zeros(max_rec + 1)
    rec_cutoffs = np.zeros(max_rec + 1)

    def dec(u):
        x = env.get_pos_vel()
        sim.set_pos_vel(*x)
        delta_s, _, _, _ = sim.step(u)
        sim.set_pos_vel(*x)
        return ocs[t].decision_function([delta_s])[0,0]

    
    while j < max_rec and score_last < (KLs[t] * np.linalg.norm(a_intended)) and score_last > 0:
        rec_info['triggered'] = True
        rec_scores[j] = score_last
        rec_cutoffs[j] = np.linalg.norm(a_intended) * KLs[t]

        delta = max(score_last, 0.0) / KLs[t]
        delta_u = 10 * utils.finite_diff1(np.zeros(d), dec)
        if np.linalg.norm(delta_u) * KLs[t] > score_last:
            delta_u = delta * delta_u / np.linalg.norm(delta_u) / 5.0
        u_r = delta * delta_u

        s, _, _, _ = env.step(u_r)
        if visualize:
            env.render()
        
        score_last = ocs[t].decision_function([s])[0,0]

        a_intended = agent.intended_action(s)
        j += 1


    rec_scores[j] = score_last
    rec_scores[j+1:] = score_last
    rec_cutoffs[j:] = KLs[t] * np.linalg.norm(a_intended)

    if rec_info['triggered']:
        print "\t\tRecovery was activated, stopped at t: " + str(t) + ", j: " + str(j)


    if j == max_rec:
        rec_info['reached_max'] = True

    rec_info['score_last'] = score_last
    rec_info['rec_scores'] = rec_scores
    rec_info['rec_cutoffs'] = rec_cutoffs
    rec_info['a_intended'] = a_intended
    return s, score_last, rec_info







def no_rec_loop(s, t, score, env, sim, agent, ocs, T, opt, KLs, visualize=False, max_rec = 500):
    rec_scores = np.zeros(max_rec + 1)
    rec_cutoffs = np.zeros(max_rec + 1)
    rec_info = {'reached_max': False, 'triggered': False}
    rec_info['score_last'] = score
    rec_info['rec_scores'] = rec_scores
    rec_info['rec_cutoffs'] = rec_cutoffs
    rec_info['a_intended'] = agent.intended_action(s)
    if score < (KLs[t] * np.linalg.norm(rec_info['a_intended'])):
        rec_info['reached_max'] = True

    return s, score, rec_info





































def collect_rec_approx_grad(env, agent, ocs, T, opt, KLs, visualize=False, early_stop=True, init_state=None, max_rec=max_rec):
    states = []
    intended_actions = []
    taken_actions = []
    scores = []
    mags = []

    s = env.reset()
    if init_state:
        env.set_pos_vel(*init_state)
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
    info['triggered'] = False

    info['initial_state'] = env.get_pos_vel()

    info['completed'] = False
    info['failed'] = False
    info['failed_in_support'] = False

    if ocs[0].predict([s])[0] == -1:
        print "Initial state predicted out of distribution"
        reject = True

    else:
        for t in range(T):
            triggered = False
            score = ocs[t].decision_function([s])[0, 0]
            score_last = score

            if score < 0.0 and info['first_out'] == -1:
                info['first_out'] = t

            failed = env.violation()
            if failed and info['first_violation'] == -1:
                info['failed'] = True
                if info['first_out'] == -1:
                    info['failed_in_support'] = True
                info['first_violation'] = t

            if failed or score < 0.0:
                break

            completed = env.completed()
            if completed and info['first_complete'] == -1:
                info['completed'] = True
                info['first_complete'] = t
                break

            a_intended = agent.intended_action(s)

            j = 0
            if not info['triggered']:
                rec_scores = np.zeros(max_rec+1)
                rec_cutoffs = np.zeros(max_rec+1)
            while j < max_rec and score_last < (KLs[t] * np.linalg.norm(a_intended)) and score_last > 0:
                if not info['triggered']:
                    rec_scores[j] = score_last
                    rec_cutoffs[j] = (KLs[t] * np.linalg.norm(a_intended))
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

                eta = np.abs(delta_score) / 50.0 / KLs[t]
                if delta <= 0:
                    u_r = np.zeros(true_delta_u.shape)
                else:
                    u_r = eta * true_delta_u / np.linalg.norm(true_delta_u)

                s, _, _, _ = env.step(u_r)
                if visualize:
                    env.render()
                score_last = ocs[t].decision_function([s])[0, 0]

                a_intended = agent.intended_action(s)
                # print "time step: " + str(t) + ", j: " + str(j)
                # print "learner norm: " + str(np.linalg.norm(a_intended))
                # print "cutoff: " + str(KLs[t] * np.linalg.norm(a_intended))
                # print "score last: " + str(score_last)
                # print "Completed: " + str(env.completed())
                # print "Failure: " + str(env.violation())
                # print "\n"
                j += 1
            
            if not info['triggered']:
                rec_scores[j] = score_last
                rec_scores[j+1:] = score_last 
                rec_cutoffs[j:] = KLs[t] * np.linalg.norm(a_intended)

            if triggered:
                print "\t\tRecovery was activated, stopped at t: " + str(t) + ", j: " + str(j)
                info['triggered'] = True

            if j == max_rec:
                print "\t\tNot able to recover, stopping"
                break


            score_updated = score_last
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
        info['rec_scores'] = rec_scores
        info['rec_cutoffs'] = rec_cutoffs
    return states, intended_actions, taken_actions, reward, freq, scores, mags, info, failed, reject



def collect_rec_random(env, agent, ocs, T, opt, KLs, visualize=False, early_stop=True, init_state=None, max_rec=max_rec):
    states = []
    intended_actions = []
    taken_actions = []
    scores = []
    mags = []

    s = env.reset()
    if init_state:
        env.set_pos_vel(*init_state)
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
    info['triggered'] = False

    info['initial_state'] = env.get_pos_vel()
    
    info['completed'] = False
    info['failed'] = False
    info['failed_in_support'] = False


    if ocs[0].predict([s])[0] == -1:
        print "Initial state predicted out of distribution"
        reject = True

    else:
        for t in range(T):
            triggered = False
            score = ocs[t].decision_function([s])[0, 0]
            score_last = score

            if score < 0.0 and info['first_out'] == -1:
                info['first_out'] = t

            failed = env.violation()
            if failed and info['first_violation'] == -1:
                info['failed'] = True
                if info['first_out'] == -1:
                    info['failed_in_support'] = True
                info['first_violation'] = t

            if failed or score < 0.0:
                break

            completed = env.completed()
            if completed and info['first_complete'] == -1:
                info['completed'] = True
                info['first_complete'] = t
                break

            a_intended = agent.intended_action(s)


            j = 0
            if not info['triggered']:
                rec_scores = np.zeros(max_rec+1)
                rec_cutoffs = np.zeros(max_rec+1)
            while j < max_rec and score_last < (KLs[t] * np.linalg.norm(a_intended)) and score_last > 0:
                if not info['triggered']:
                    rec_scores[j] = score_last
                    rec_cutoffs[j] = (KLs[t] * np.linalg.norm(a_intended))
                triggered = True

                delta_u = np.random.normal(0, .01, env.action_space.shape)
                delta = max(score_last, 0.0) / 500.0 / KLs[t]
                delta_u = delta * delta_u / np.linalg.norm(delta_u)

                delta_x, _, _, _ = env.step(delta_u)
                delta_score = ocs[t].decision_function([delta_x])[0, 0]
                if delta_score < score_last:
                    delta_u = - delta_u
                if delta_score == score_last:
                    delta_u = np.zeros(d)

                eta = np.abs(delta_score) / 50.0 / KLs[t]
                if delta <= 0:
                    u_r = np.zeros(delta_u.shape)
                else:
                    u_r = eta * delta_u / delta

                s, _, _, _ = env.step(u_r)
                if visualize:
                    env.render()
                score_last = ocs[t].decision_function([s])[0, 0]

                a_intended = agent.intended_action(s)
                # print "time step: " + str(t) + ", j: " + str(j)
                # print "learner norm: " + str(np.linalg.norm(a_intended))
                # print "cutoff: " + str(KLs[t] * np.linalg.norm(a_intended))
                # print "score last: " + str(score_last)
                # print "Completed: " + str(env.completed())
                # print "Failure: " + str(env.violation())
                # print "\n"
                j += 1

            if not info['triggered']:
                rec_scores[j] = score_last
                rec_scores[j+1:] = score_last 
                rec_cutoffs[j:] = KLs[t] * np.linalg.norm(a_intended)

            if triggered:
                print "\t\tRecovery was activated, stopped at t: " + str(t) + ", j: " + str(j)
                info['triggered'] = True

            if j == max_rec:
                print "\t\tNot able to recover, stopping"
                break

            score_updated = score_last

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
        info['rec_scores'] = rec_scores
        info['rec_cutoffs'] = rec_cutoffs
    return states, intended_actions, taken_actions, reward, freq, scores, mags, info, failed, reject



def collect_rec_finite_diff(env, sim, agent, ocs, T, opt, KLs, visualize=False, early_stop=True, init_state=None, max_rec=max_rec):
    states = []
    intended_actions = []
    taken_actions = []
    scores = []
    mags = []

    s = env.reset()
    if init_state:
        env.set_pos_vel(*init_state)
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
    info['triggered'] = False

    info['initial_state'] = env.get_pos_vel()

    info['completed'] = False
    info['failed'] = False
    info['failed_in_support'] = False

    if ocs[0].predict([s])[0] == -1:
        print "Initial state predicted out of distribution"
        reject = True

    else:
        for t in range(T):
            def dec(u):
                x = env.get_pos_vel()
                sim.set_pos_vel(*x)
                delta_s, _, _, _ = sim.step(u)
                sim.set_pos_vel(*x)
                return ocs[t].decision_function([delta_s])[0,0]
            
            triggered = False
            score = ocs[t].decision_function([s])[0, 0]
            score_last = score

            if score < 0.0 and info['first_out'] == -1:
                info['first_out'] = t

            failed = env.violation()
            if failed and info['first_violation'] == -1:
                info['failed'] = True
                if info['first_out'] == -1:
                    info['failed_in_support'] = True
                info['first_violation'] = t

            if failed or score < 0.0:
                break

            completed = env.completed()
            if completed and info['first_complete'] == -1:
                info['completed'] = True
                info['first_complete'] = t
                break

            a_intended = agent.intended_action(s)

            j = 0
            if not info['triggered']:
                rec_scores = np.zeros(max_rec+1)
                rec_cutoffs = np.zeros(max_rec+1)
            while j < max_rec and score_last < (KLs[t] * np.linalg.norm(a_intended)) and score_last > 0:
                if not info['triggered']:
                    rec_scores[j] = score_last
                    rec_cutoffs[j] = (KLs[t] * np.linalg.norm(a_intended))
                triggered = True

                delta = max(score_last, 0.0) / KLs[t]
                delta_u = 10 * utils.finite_diff1(np.zeros(d), dec)
                if np.linalg.norm(delta_u) * KLs[t] > score_last:
                    delta_u = delta * delta_u / np.linalg.norm(delta_u) / 5.0
                u_r = delta * delta_u

                s, _, _, _ = env.step(u_r)
                if visualize:
                    env.render()
                
                score_last = ocs[t].decision_function([s])[0,0]

                a_intended = agent.intended_action(s)
                # print "time step: " + str(t) + ", j: " + str(j)
                # print "learner norm: " + str(np.linalg.norm(a_intended))
                # print "cutoff: " + str(KLs[t] * np.linalg.norm(a_intended))
                # print "score last: " + str(score_last)
                # print "Completed: " + str(env.completed())
                # print "Failure: " + str(env.violation())
                # print "\n"
                j += 1

            if not info['triggered']:
                rec_scores[j] = score_last
                rec_scores[j+1:] = score_last 
                rec_cutoffs[j:] = KLs[t] * np.linalg.norm(a_intended)

            if triggered:
                print "\t\tRecovery was activated, stopped at t: " + str(t) + ", j: " + str(j)
                info['triggered'] = True

            if j == max_rec:
                print "\t\tNot able to recover, stopping"
                break

            score_updated = score_last

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
        info['rec_scores'] = rec_scores
        info['rec_cutoffs'] = rec_cutoffs
    return states, intended_actions, taken_actions, reward, freq, scores, mags, info, failed, reject



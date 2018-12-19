import statistics
import utils
import numpy as np
from numba import jit
from scipy.linalg import inv
from utils import mul
import time as timer
import IPython

class Controller():
    def __init__(self):
        return

    def backups(self, env, T, Cs, Fs, cs, fs):
        print "Computing backwards pass"
        start_time = timer.time()

        n = env.state_dim
        d = env.action_dim

        C1, C2, C3, C4 = utils.breakup_C(Cs[-1], n)
        c1, c2 = utils.breakup_c(cs[-1], n)

        K_T = - np.dot( np.linalg.inv(C4), C3)
        k_T = - np.dot( np.linalg.inv(C4), c2)
        

        V_T = C1 + mul(C2, K_T) + mul(K_T.T, C3) + mul(K_T.T, C4, K_T)
        v_T = c1 + mul(C2, k_T) + mul(K_T.T, c2) + mul(K_T.T, C4, k_T)

        Vs, vs, Ks, ks = [None] * T, [None] * T, [None] * T, [None] * T
        Vs[-1], vs[-1], Ks[-1], ks[-1] = V_T, v_T, K_T, k_T

        for t in range(T - 1)[::-1]:
            C_t = Cs[t]
            F_t = Fs[t]
            c_t = cs[t]
            f_t = fs[t]

            # IPython.embed()
            Q_t = C_t + mul(F_t.T, Vs[t + 1], F_t)
            q_t = c_t + mul(F_t.T, Vs[t + 1], f_t) + mul(F_t.T, vs[t + 1])


            Q1, Q2, Q3, Q4 = utils.breakup_C(Q_t, n)
            q1, q2 = utils.breakup_c(q_t, n)
            Ks[t] = - mul(inv(Q4), Q3)
            ks[t] = - mul(inv(Q4), q2)
            Vs[t] = Q1 + mul(Q2, Ks[t]) + mul(Ks[t].T, Q3) + mul(Ks[t].T, Q4, Ks[t])
            vs[t] = q1 + mul(Q2, ks[t]) + mul(Ks[t].T, q2) + mul(Ks[t].T, Q4, ks[t])

        self.Vs = Vs
        self.vs = vs
        self.Ks = Ks
        self.ks = ks
        print "Done computing back pass"
        end_time = timer.time()
        print "Total time: " + str(end_time - start_time)
        return Vs, vs, Ks, ks



    @jit
    def linearize(self, env, sim, x_array, u_array):
        print "Linearizing"
        start_time = timer.time()
        aug_n = u_array[0].shape[0] + x_array[0].shape[0]
        T = len(x_array)
        Cs = [np.identity(aug_n)] * (T - 1)
        Fs = []
        cs = [np.zeros(aug_n)] * (T - 1)
        fs = [np.zeros(x_array[0].shape)] * (T - 1)
        simulate_fn = utils.gen_simulate_step(sim)
        for t in range(T - 1):
            print "linear t: " + str(t)
            x, u = x_array[t], u_array[t]
            xu = np.hstack((x, u))
            F = utils.finite_diff1(xu, simulate_fn)
            Fs.append(F)

        x = x_array[-1]
        u = np.zeros(u.shape)
        xu = np.hstack((x, u))
        Cs.append(np.identity(aug_n))
        cs.append(np.zeros(aug_n))
        Fs.append(np.zeros(Fs[0].shape))
        fs.append(np.zeros(x.shape))

        print "Done linearizing"
        end_time = timer.time()
        print "Total time: " + str(end_time - start_time)

        self.Cs = Cs
        self.Fs = Fs
        self.cs = cs
        self.fs = fs

        return Cs, Fs, cs, fs


    @jit
    def contr(self, x, t, x_ref=0, u_ref=0):
        contr = mul(self.Ks[t], x - x_ref) + self.ks[t] + u_ref
        return contr





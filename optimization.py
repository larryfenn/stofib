from numpy import random
import numpy as np
from scipy import stats
from scipy.constants import golden
from simulation import simulate
from simulation import simgradient
from collections import namedtuple
import timeit


def optimize(u_method, obs, **kwargs):
    ''' optimization function
    coords - spot where the function is evaluated
    f - result of the evaluation
    '''
    Point = namedtuple("point", ["x", "y"])
    y = kwargs['y']
    sim_n = kwargs['sim_n']
    output = dict()
    cpustart = timeit.default_timer()

    # some decision procedure that results in the x y result

    if u_method is 'golden':
        # deterministic/non evaluating parts
        simtime = 0
        alpha = kwargs['alpha']
        epsilon = kwargs['epsilon']
        a = kwargs['a']
        b = kwargs['b']
        output['a'] = a
        output['b'] = b
        c = b + (a - b)/golden
        ob_c = Point(x=c, y=y)
        d = a + (b - a)/golden
        ob_d = Point(x=d, y=y)
        pick = None

        # sample and confidence loop
        # 1. check if either f(c) or f(d) exists. if not, create them.
        #    2 samples are needed to establish a CI
        #    if they do exist, sample only from the larger CI
        explored = True
        if ob_c not in obs:
            explored = False
            csim = simulate(c, y, sim_n=sim_n)
            simtime += csim['simtime']
            obs[ob_c] = [csim['f']]
            csim = simulate(c, y, sim_n=sim_n)
            simtime += csim['simtime']
            obs[ob_c].append(csim['f'])
        if ob_d not in obs:
            explored = False
            dsim = simulate(d, y, sim_n=sim_n)
            simtime += dsim['simtime']
            obs[ob_d] = [dsim['f']]
            dsim = simulate(d, y, sim_n=sim_n)
            simtime += dsim['simtime']
            obs[ob_d].append(dsim['f'])
        # Compute confidence intervals (prerequisite for future steps)
        ci_c = stats.norm.interval(
            alpha, loc=np.mean(obs[ob_c]),
            scale=np.std(obs[ob_c], ddof=1)/np.sqrt(len(obs[ob_c])))
        ci_d = stats.norm.interval(
            alpha, loc=np.mean(obs[ob_d]),
            scale=np.std(obs[ob_d], ddof=1)/np.sqrt(len(obs[ob_d])))
        if explored:
            if (ci_c[1] - ci_c[0]) > (ci_d[1] - ci_d[0]):
                csim = simulate(c, y, sim_n=sim_n)
                simtime += csim['simtime']
                obs[ob_c].append(csim['f'])
                ci_c = stats.norm.interval(
                    alpha, loc=np.mean(obs[ob_c]),
                    scale=np.std(obs[ob_c], ddof=1)/np.sqrt(len(obs[ob_c])))
            else:
                dsim = simulate(d, y, sim_n=sim_n)
                simtime += dsim['simtime']
                obs[ob_d].append(dsim['f'])
                ci_d = stats.norm.interval(
                    alpha, loc=np.mean(obs[ob_d]),
                    scale=np.std(obs[ob_d], ddof=1)/np.sqrt(len(obs[ob_d])))

        # 2. while confidence intervals are not disjoint, keep shrinking em
        while (ci_c[1] >= ci_d[0]) and (ci_d[1] >= ci_c[0]):
            # exit condition: both CIs are so small we are indifferent
            if max(ci_c[1] - ci_c[0], ci_d[1] - ci_d[0]) < epsilon:
                break
            if (ci_c[1] - ci_c[0]) > (ci_d[1] - ci_d[0]):
                csim = simulate(c, y, sim_n=sim_n)
                simtime += csim['simtime']
                obs[ob_c].append(csim['f'])
                ci_c = stats.norm.interval(
                    alpha, loc=np.mean(obs[ob_c]),
                    scale=np.std(obs[ob_c], ddof=1)/np.sqrt(len(obs[ob_c])))
            else:
                dsim = simulate(d, y, sim_n=sim_n)
                simtime += dsim['simtime']
                obs[ob_d].append(dsim['f'])
                ci_d = stats.norm.interval(
                    alpha, loc=np.mean(obs[ob_d]),
                    scale=np.std(obs[ob_d], ddof=1)/np.sqrt(len(obs[ob_d])))
        # 3. now that CIs are disjoint, take the lower one
        # case: confidence interval at c is fully below interval at d
        if ci_c[1] < ci_d[0]:
            pick = 'c'
        # case: confidence interval at d is fully below interval at c
        if ci_d[1] < ci_c[0]:
            pick = 'd'
        if pick is 'c':
            output['b'] = d
        else:
            output['a'] = c
        output['f_c'] = np.mean(obs[ob_c])
        output['f_d'] = np.mean(obs[ob_d])

        # end loop

        output['simtime'] = simtime

    # ***

    if u_method is 'gradient':
        simout = simgradient(
            kwargs['x'], kwargs['y'], sim_grad_n=kwargs['sim_grad_n'])
        output['gradient'] = simout['gradient']
        output['x'] = kwargs['x'] - kwargs['stepsize'] * output['gradient']
        output['simtime'] = simout['simtime']
        simout = simulate(
            output['x'], kwargs['y'], sim_n=sim_n)
        output['f'] = simout['f']
        output['simtime'] += simout['simtime']

    # ***

    # end decision procedure

    output['opttime'] = timeit.default_timer() - cpustart - output['simtime']
    return output

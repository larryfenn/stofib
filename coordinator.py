from optimization import optimize
import timeit
import itertools

# casual notes: gradient actually has a lot of problem minimizing functions that get flat in regions.


def checkCond(u_method, **kwargs):
    output = dict(stop=False)
    cpustart = timeit.default_timer()

#    print kwargs

    if u_method is 'gradient':
        if kwargs['f'] == -1 or kwargs['gradient'] == -1:
            output['stop'] = True
            output['result'] = False
        elif kwargs['f'] < kwargs['constraint']:
            output['stop'] = True
            output['result'] = True
        elif abs(kwargs['gradient']) < kwargs['epsilon']:
            output['stop'] = True
            output['result'] = False

    if u_method is 'golden':
        if ((kwargs['f_c'] < kwargs['constraint']) and
            (kwargs['f_d'] < kwargs['constraint'])):
            output['stop'] = True
            output['result'] = True
        elif kwargs['b'] - kwargs['a'] < kwargs['delta']:
            output['stop'] = True
            output['result'] = False

    output['time'] = timeit.default_timer() - cpustart
    return output


# bayesian median finding helper
def find_median(y0, y1, pdf):
    for i in range(y0, y1 + 1):
        if sum(pdf[:i+1]) >= .5:
            break
    return i


def trial(**kwargs):
    # program tracker vars
    simtime = 0
    opttime = 0
    btime = 0
    condtime = 0
    itercount = 0
    final = dict()

    # problem specific vars/function related
    init_x = .5
    init_a = 0
    init_b = 1
    constraint = 3
    y0 = 65
    y1 = 97

    # bayesian discrete probability structure
    pdf = [0] * (y1 + 1)
    for i in range(y0, y1 + 1):
        pdf[i] = 1./(y1 - y0 + 1)

    # bisection search history data structure
    biobs = list()

    # golden search history data structure
    obs = dict()  # maps Point objects to a list of observations at that Point

    # THE FOLLOWING ARE ALL HYPERPARAMETERS
    # in common:
    sim_n = 100
    sim_n = kwargs['sim_n']

    # golden search
    # these work for good precision
    alpha = .9
    alpha = kwargs['alpha']
    gs_epsilon = .001
    gs_epsilon = kwargs['gs_epsilon']
    delta = .01
    delta = kwargs['delta']

    # gradient
    sim_grad_n = 5000
    sim_grad_n = kwargs['sim_grad_n']
    grad_epsilon = .01
    grad_epsilon = kwargs['grad_epsilon']
    stepsize = .1
    stepsize = kwargs['stepsize']

    # bisect
    b_error = .1
    b_error = kwargs['b_error']

    # bayesian
    oracle = .6
    oracle = kwargs['oracle']
    certainty = .99
    certainty = kwargs['certainty']

    # control logic vars
    u_method = 'golden'
    u_method = 'gradient'
    u_method = kwargs['u_method']
    b_method = 'bayes'
    b_method = 'bisect'
    b_method = kwargs['b_method']

    # END HYPERPARAMETERS

    # initializing vars
    outer_stop = False
    left = y0
    right = y1
    error = b_error
    cpustart = timeit.default_timer()
    itercount = 0
    old_y = 0
    bayescount = 0
    while not outer_stop:
        if b_method is 'bayes':
            y = find_median(y0, y1, pdf)
            if y == old_y:
                bayescount += 1
            else:
                bayescount = 0
            old_y = y
        if b_method is 'bisect':
            alpha = 1 - error
            y = (left + right)/2
        x = init_x
        a = init_a
        b = init_b
        stop = False
        while not stop:
            itercount += 1
            if u_method is 'gradient':
                optresult = optimize(u_method=u_method,
                                     obs=obs,
                                     stepsize=stepsize,
                                     constraint=constraint,
                                     x=x,
                                     y=y,
                                     sim_n=sim_n,
                                     sim_grad_n=sim_grad_n)
                x = optresult['x']
                cond = checkCond(u_method=u_method,
                                 constraint=constraint,
                                 epsilon=grad_epsilon,
                                 f=optresult['f'],
                                 gradient=optresult['gradient'])
                opttime += optresult['opttime']
                simtime += optresult['simtime']

            if u_method is 'golden':
                optresult = optimize(u_method=u_method,
                                     obs=obs,
                                     alpha=alpha,
                                     epsilon=gs_epsilon,
                                     sim_n=sim_n,
                                     a=a,
                                     b=b,
                                     y=y)
                a = optresult['a']
                b = optresult['b']
                f_c = optresult['f_c']
                f_d = optresult['f_d']
                opttime += optresult['opttime']
                simtime += optresult['simtime']
                cond = checkCond(u_method=u_method,
                                 a=a,
                                 b=b,
                                 delta=delta,
                                 constraint=constraint,
                                 f_c=f_c,
                                 f_d=f_d)
            stop = cond['stop']
            condtime += cond['time']

        if u_method is 'golden':
            print str(y) + ": " + str(f_c) + " " + str(f_d) + " " + str(a) + " " + str(b)
        if u_method is 'gradient':
            print str(y) + ": " + str(optresult['f']) + " " + str(x)

        print cond['result']
        # ***

        if b_method is 'bayes':
            # update priors
            if cond['result']:
                for i in range(y0, y):
                    pdf[i] *= (1 - oracle)
                for i in range(y, y1 + 1):
                    pdf[i] *= oracle
            else:
                for i in range(y0, y + 1):
                    pdf[i] *= oracle
                for i in range(y + 1, y1 + 1):
                    pdf[i] *= (1-oracle)
            # rescale
            scalefactor = 1./sum(pdf)
            for i in range(y0, y1 + 1):
                pdf[i] *= scalefactor
#            print max(pdf)
            if max(pdf) > certainty or bayescount > 25:
                outer_stop = True
                final['y'] = pdf.index(max(pdf))

        # ***

        if b_method is 'bisect':
            biobs.append(dict(left=left, y=y, right=right, result=cond['result'], error=error))
            # check for backtracking here
            # 1. identify the current T/F streak length
            streaklen = 0
            for i in range(1, len(biobs) + 1):
                if biobs[-i]['result'] == cond['result']:
                    streaklen += 1
                else:
                    break
            root = max(0, len(biobs) - streaklen)  # possible mistake index
            # 2. compare pcs/confidence for this streak with the undo probability
            backtrack = biobs[root]['error'] > ((.5 ** (streaklen - 1)) * b_error)
#            print "backtrack: " + str(backtrack)
            if backtrack:
                left = biobs[root]['left']
                right = biobs[root]['right']
                error = biobs[root]['error'] / 2
                # prune the list back to before the mistake; drop the items
                # GS values are still cached in the obs data structure
                for i in range(len(biobs) - root):
                    del biobs[-1]
            else:
                error = b_error
                if cond['result']:
                    # discard left
                    left = y
                else:
                    right = y
                if right - left == 1:
                    outer_stop = True
                    final['y'] = left
    btime = timeit.default_timer() - cpustart
    btime = btime - simtime - opttime - condtime
    final['itercount'] = itercount
    final['simtime'] = simtime
    final['optime'] = opttime
    final['condtime'] = condtime
    final['btime'] = btime
    return final


# TESTS:
# 1. Gradient Search + Bayesian
# 2. Gradient Search + Bisect
# 3. Golden section + Bayesian
# 4. Golden section + Bisect
range_sim_n = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
range_alpha = [.6, .75, .9]
range_gs_epsilon = [.1, .01, .001]
range_delta = [.1, .01]
range_sim_grad_n = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
range_grad_epsilon = [.1, .01, .001]
range_stepsize = [.5, .1, .05]
range_b_error = [.1, .2, .3]
range_oracle = [.6, .7, .8, .9]
range_certainty = [.7, .9]
range_u_method = ['golden', 'gradient']
range_b_method = ['bisect', 'bayes']

with open('records.csv', 'wb') as f:
    f.write("u_method,b_method,solution,iterations,simtime,optime,condtime,btime,sim_n,alpha,gs_epsilon,delta,sim_grad_n,grad_epsilon,stepsize,b_error,oracle,certainty\n")
    for sim_n, alpha, gs_epsilon, delta, sim_grad_n, grad_epsilon, stepsize, b_error, oracle, certainty, u_method, b_method in itertools.product(range_sim_n, range_alpha, range_gs_epsilon, range_delta, range_sim_grad_n, range_grad_epsilon, range_stepsize, range_b_error, range_oracle, range_certainty, range_u_method, range_b_method):
        for i in range(10):
            print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
            rowdata = trial(sim_n=sim_n,
                            alpha=alpha,
                            gs_epsilon=gs_epsilon,
                            delta=delta,
                            sim_grad_n=sim_grad_n,
                            grad_epsilon=grad_epsilon,
                            stepsize=stepsize,
                            b_error=b_error,
                            oracle=oracle,
                            certainty=certainty,
                            u_method=u_method,
                            b_method=b_method)
            f.write(u_method + ',' +
                    b_method + ',' +
                    str(rowdata['y']) + ',' +
                    str(rowdata['itercount']) + ',' +
                    str(rowdata['simtime']) + ',' +
                    str(rowdata['optime']) + ',' +
                    str(rowdata['condtime']) + ',' +
                    str(rowdata['btime']) + ',' +
                    str(sim_n) + ',' +
                    str(alpha) + ',' +
                    str(gs_epsilon) + ',' +
                    str(delta) + ',' +
                    str(sim_grad_n) + ',' +
                    str(grad_epsilon) + ',' +
                    str(stepsize) + ',' +
                    str(b_error) + ',' +
                    str(oracle) + ',' +
                    str(certainty) + '\n')












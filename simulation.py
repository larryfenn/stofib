from numpy import random
import math
import timeit
random.seed(1)


# true 'right' answer is y=76, x = whatever
def simulate(x, y, **kwargs):
    output = dict()
    cpustart = timeit.default_timer()

    # put in function evaluation stuff here.

    solution = 0
    for i in range(kwargs['sim_n']):
        sample = random.exponential(1/x)
        solution += .01*y + math.log(sample + 1) + 4/(2*sample + 1)
    solution /= kwargs['sim_n']

    # end function definition space

    output['f'] = (solution)
    output['simtime'] = timeit.default_timer() - cpustart
    return output


def simgradient(x, y, **kwargs):
    output = dict()
    cpustart = timeit.default_timer()

    # put in gradient evaluation
    solution = 0
    for i in range(kwargs['sim_grad_n']):
        sample = random.exponential(1/x)
        solution += (sample/x) * ((1/(sample + 1)) - (8/(2*sample + 1)**2))
    solution /= kwargs['sim_grad_n']

    # end gradient definition space

    output['gradient'] = (solution)
    output['simtime'] = timeit.default_timer() - cpustart
    return output

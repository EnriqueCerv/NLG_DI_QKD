#%%
import ncpol2sdpa
import numpy as np
from sympy import *
from ncpol2sdpa import SdpRelaxation, generate_measurements
import os

# %%

'''
This file calculates the tripartite winning probability of the CHSH game from Section III B of 
Device independent security of quantum key distribution from arbitrary monogamy-of-entanglement games

Includes the constrained tripartite winning probabilities given Alice and Bob expected score
'''


# %%

os.mkdir('CHSH__Data')

# # # # # # # # # # # # # # # # # # # # # # # # 
# Get projection operators and substitution
# # # # # # # # # # # # # # # # # # # # # # # # 

def PVM():
    P = ncpol2sdpa.generate_measurements([2]*2, 'P')
    Q = ncpol2sdpa.generate_measurements([2]*2, 'Q')
    F = ncpol2sdpa.generate_measurements([2]*4, 'F')

    return P, Q, F

def PVM_substitutions(P,Q,F):
    substitutions = ncpol2sdpa.projective_measurement_constraints((P,Q,F))

    return substitutions


# # # # # # # # # # # # # # # # # # # # # # # # 
# Objective functions
# # # # # # # # # # # # # # # # # # # # # # # # 

'''Objective function for the three players'''
def objective_function(P,Q,F):
    zero_zero = P[0][0] * Q[0][0] * F[0][0] + (1 - P[0][0]) * (1 - Q[0][0]) * (1 - F[0][0])
    zero_one = P[0][0] * Q[1][0] * F[1][0] + (1 - P[0][0]) * (1 - Q[1][0]) * (1 - F[1][0])
    one_zero = P[1][0] * Q[0][0] * F[2][0] + (1 - P[1][0]) * (1 - Q[0][0]) * (1 - F[2][0])
    one_one =  P[1][0] * (1 - Q[1][0]) * F[3][0] + (1 - P[1][0]) * Q[1][0] * (1 - F[3][0])
    return (-zero_zero - zero_one - one_zero -one_one)/4


'''Objective function for Alice and Bob'''
def objective_function_AB(P,Q,eps):
    constraint = -3/4 - eps
    # constraint = eps - np.cos(np.pi/8)**2
    x_zero = P[0][0] * (Q[0][0] + Q[1][0]) + (1 - P[0][0]) * ((1 - Q[0][0]) + (1 - Q[1][0]))
    x_one = P[1][0] * (Q[0][0] + (1 - Q[1][0])) + (1 - P[1][0]) * ((1 - Q[0][0]) + Q[1][0])

    return constraint + (x_zero + x_one)/4


# # # # # # # # # # # # # # # # # # # # # # # # 
#   We get the relaxation on the desired level
# # # # # # # # # # # # # # # # # # # # # # # # 

def relaxation(level, eps, verbose=1, save=True):
    
    if save:
        dir = 'CHSH__Data/CHSH_data_for_level_%s' %level
        if eps is not None:    
            dir += '_epsAB_equal_%s' %eps
        os.mkdir(dir)
    
    '''The variables'''
    variables = PVM()
    P = variables[0]
    Q = variables[1]
    F = variables[2]
    variables = ncpol2sdpa.flatten([P,Q,F])

    '''The substitions'''
    substitutions = PVM_substitutions(P,Q,F)

    '''The constraints'''
    inequalities = [objective_function_AB(P,Q,eps)]

    '''The objective function'''
    objective = objective_function(P,Q,F)

    '''Getting the relaxation'''
    sdpRelaxation = SdpRelaxation(variables=variables, verbose = verbose)
    sdpRelaxation.get_relaxation(level, objective=objective, 
                substitutions=substitutions,
                momentinequalities=inequalities)
    sdpRelaxation.solve(solver='mosek')

    if save:
        '''Getting the solved parameters'''
        primal_mat = sdpRelaxation.x_mat[0]
        primal_val = sdpRelaxation.primal
        dual_mat = sdpRelaxation.y_mat[0]
        dual_val = sdpRelaxation.dual
        status = sdpRelaxation.status
        time = sdpRelaxation.solution_time
        monomial_sets = sdpRelaxation.monomial_sets[0]

        '''Saving the solved parameters'''
        np.savetxt(dir + '/primal_mat.csv', primal_mat)
        np.savetxt(dir + '/dual_mat.csv', dual_mat)
        np.save(dir + '/monomial_sets', monomial_sets)
        sdpRelaxation.save_monomial_index(dir + '/monomial_index.txt')
        primal_string = 'The primal objective value is %s. This is an upper bound on the CHSH quantum value.' %(-primal_val)
        dual_string = '\n The dual objective value is %s.' %(-dual_val)
        stat_string = '\n The status of level %s' %level 
        stat_string += ' is ' + status 
        stat_string += ' and it required %s' %time 
        stat_string += ' seconds.'
        str_output = primal_string+dual_string+stat_string
        output_file = open(dir + '/output_lvl%s.txt' %level, 'wt')
        n = output_file.write(str_output)
        output_file.close()
        # print(str_output)

    print('The optimum of level ', level, 
            'with AB winning probability at least ', eps+3/4,
            ' is: ', -sdpRelaxation.primal)
    print('The sdp: ', sdpRelaxation.status)
    
    "The following are returned for making a figure"

    return -sdpRelaxation.primal, 3/4+eps

# %%

# # # # # # # # # # # # # # # # # # # # # # # # 
#   Collecting data
# # # # # # # # # # # # # # # # # # # # # # # # 

ABC_win = []
AB_win = []
for eps in range(0,101,5):
    out = relaxation(level=3, eps=eps/1000,verbose=0)
    ABC_win.append(out[0])
    AB_win.append(out[1])


out = relaxation(level=3, eps=(np.sqrt(2)+2)/4-3/4,verbose=0)
ABC_win.append(out[0])
AB_win.append(out[1])

np.savetxt('CHSH__Data/Pr_ABwin', AB_win)
np.savetxt('CHSH__Data/Pr_ABCwin', ABC_win)

import matplotlib.pyplot as plt

plt.scatter(AB_win, ABC_win, alpha=0.5)
plt.plot(AB_win, [(np.sqrt(2)+2)/8]*len(AB_win))
plt.xlabel('Pr[A,B win]')
plt.ylabel('Pr[A,B,C win | A,B win]')
plt.savefig(fname='CHSH__Data/chsh_constrained.png')
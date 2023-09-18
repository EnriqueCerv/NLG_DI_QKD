#%%
import ncpol2sdpa
import numpy as np
from sympy import *
from ncpol2sdpa import SdpRelaxation, generate_measurements
import os

# %%

'''
This file calculates the tripartite winning probability of the modified CHSH game from Section III C of 
Device independent security of quantum key distribution from arbitrary monogamy-of-entanglement games

Includes the constrained tripartite winning probabilities given Alice and Bob expected score
'''


# %%
os.mkdir('CHSH_mod__Data')

# %%

# # # # # # # # # # # # # # # # # # # # # # # # 
# Get projection operators and substitution for CHSHmod_2
# # # # # # # # # # # # # # # # # # # # # # # # 

def PVM_2():
    P = ncpol2sdpa.generate_measurements([2]*3, 'P')
    Q = ncpol2sdpa.generate_measurements([2]*2, 'Q')

    return P, Q

def PVM_substitutions_2(P,Q):
    substitutions = ncpol2sdpa.projective_measurement_constraints((P,Q))

    return substitutions


# # # # # # # # # # # # # # # # # # # # # # # # 
# Objective function for CHSHmod_2
# # # # # # # # # # # # # # # # # # # # # # # # 

def objective_function_2(P,Q):
    obj = (P[0][0] * Q[0][0] + (1-P[0][0]) * (1-Q[0][0]) + \
        P[0][0] * Q[1][0] + (1-P[0][0]) * (1-Q[1][0]) + \
        P[1][0] * Q[0][0] + (1-P[1][0]) * (1-Q[0][0]) + \
        P[1][0] * (1-Q[1][0]) + (1-P[1][0]) * Q[1][0])/6

    obj += (P[2][0] * Q[0][0] + (1 - P[2][0]) * (1 - Q[0][0]) +\
        P[2][0] * Q[1][0] + (1 - P[2][0]) * (1 - Q[1][0]))/6
    return -obj


# # # # # # # # # # # # # # # # # # # # # # # # 
#   We get the relaxation on the desired level for CHSHmod_2
# # # # # # # # # # # # # # # # # # # # # # # # 

def relaxation_2(level, verbose=1):
    
    '''The variables'''
    P,Q = PVM_2()
    variables = ncpol2sdpa.flatten([P,Q])

    '''The substitions'''
    substitutions = PVM_substitutions_2(P,Q)

    '''The objective function'''
    objective = objective_function_2(P,Q)

    '''Getting the relaxation'''
    sdpRelaxation = SdpRelaxation(variables=variables, verbose = verbose)
    sdpRelaxation.get_relaxation(level, objective=objective, substitutions=substitutions)
    sdpRelaxation.solve()
    print('The optimum of level ', level, 'of the hierarchy is: ', -sdpRelaxation.primal)

    return -sdpRelaxation.primal

relaxation_2(level=3,verbose=1)
# %%

# # # # # # # # # # # # # # # # # # # # # # # # 
# Get projection operators and substitution for CHSHmod_3
# # # # # # # # # # # # # # # # # # # # # # # # 

def PVM():
    P = ncpol2sdpa.generate_measurements([2]*3, 'P')
    Q = ncpol2sdpa.generate_measurements([2]*2, 'Q')
    F = ncpol2sdpa.generate_measurements([2]*6, 'F')

    return P, Q, F

def PVM_substitutions(P,Q,F):
    substitutions = ncpol2sdpa.projective_measurement_constraints((P,Q,F))

    return substitutions


# # # # # # # # # # # # # # # # # # # # # # # # 
# Objective functions or CHSHmod_3
# # # # # # # # # # # # # # # # # # # # # # # # 

'''Objective function for the three players'''
def objective_function(P,Q,F):
    obj = (P[0][0] * Q[0][0] * F[0][0] + (1 - P[0][0]) * (1 - Q[0][0]) * (1 - F[0][0]) + \
        P[0][0] * Q[1][0] * F[1][0] + (1 - P[0][0]) * (1 - Q[1][0]) * (1 - F[1][0]) + \
        P[1][0] * Q[0][0] * F[2][0] + (1 - P[1][0]) * (1 - Q[0][0]) * (1 - F[2][0]) + \
        P[1][0] * (1 - Q[1][0]) * F[3][0] + (1 - P[1][0]) * Q[1][0] * (1 - F[3][0]))/6

    obj += (P[2][0] * Q[0][0] * F[4][0] + (1 - P[2][0]) * (1 - Q[0][0]) * (1 - F[4][0]) +\
        P[2][0] * Q[1][0] * F[5][0] + (1 - P[2][0]) * (1 - Q[1][0]) * (1 - F[5][0]))/6

    return -obj


'''Objective function for Alice and Bob'''
def objective_function_AB(P,Q,score):
    constraint = -score
    obj = (P[0][0] * Q[0][0] + (1-P[0][0]) * (1-Q[0][0]) + \
        P[0][0] * Q[1][0] + (1-P[0][0]) * (1-Q[1][0]) + \
        P[1][0] * Q[0][0] + (1-P[1][0]) * (1-Q[0][0]) + \
        P[1][0] * (1-Q[1][0]) + (1-P[1][0]) * Q[1][0])/6

    obj += (P[2][0] * Q[0][0] + (1 - P[2][0]) * (1 - Q[0][0]) +\
        P[2][0] * Q[1][0] + (1 - P[2][0]) * (1 - Q[1][0]))/6

    return constraint + obj


# # # # # # # # # # # # # # # # # # # # # # # # 
#   We get the relaxation on the desired level
# # # # # # # # # # # # # # # # # # # # # # # # 


def relaxation(level, score, verbose=1, save=False):
    
    if save:
        dir = 'CHSH_mod__Data/CHSH_data_for_level_%s' %level
        if score is not None:    
            dir += '_epsAB_equal_%s' %score
        os.mkdir(dir)
    
    '''The variables'''
    P,Q,F = PVM()
    variables = ncpol2sdpa.flatten([P,Q,F])

    '''The substitutions'''
    substitutions = PVM_substitutions(P,Q,F)

    '''The constraints'''
    inequalities = [objective_function_AB(P,Q, score)]

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
            'with AB winning probability at least ', score,
            ' is: ', -sdpRelaxation.primal)
    print('The sdp: ', sdpRelaxation.status)
    
    "The following are returned for making a figure"

    return -sdpRelaxation.primal, score


# relaxation(level=2, score = 0.875, verbose=1)
# %%

# # # # # # # # # # # # # # # # # # # # # # # # 
# Cons data for fully uniform CHSH_mod
# # # # # # # # # # # # # # # # # # # # # # # # 
ABC_win = []
AB_win = []
o3 = 1/2+1/3
o2 = relaxation_2(level = 2, verbose = 0)
scores = np.linspace(o3, o2, 20)

for score in scores:
    out = relaxation(level=2, score=score,verbose=0)
    ABC_win.append(out[0])
    AB_win.append(out[1])


import matplotlib.pyplot as plt

np.savetxt('CHSH_mod__Data/Pr_ABwin', AB_win)
np.savetxt('CHSH_mod__Data/Pr_ABCwin', ABC_win)

plt.scatter(AB_win, ABC_win)
plt.savefig('CHSH_mod__Data/Pr_ABC_constrained')

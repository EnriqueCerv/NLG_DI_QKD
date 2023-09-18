#%%
import ncpol2sdpa
import numpy as np
from sympy import *
from ncpol2sdpa import SdpRelaxation, generate_measurements
import os

# %%

'''
This file calculates the tripartite winning probability of the MSG from Section III A of 
Device independent security of quantum key distribution from arbitrary monogamy-of-entanglement games

Includes the quantum value of MSG_3 (optimized by setting score = 8/9, eps_AC = eps_BC = 1/9) in relaxation function, 
and the constrained tripartite winning probabilities given Alice and Bob expected score
'''


# %%
os.mkdir('MSG__Data')

# %%

# # # # # # # # # # # # # # # # # # # # # # # # 
# Get projection operators and substitution
# # # # # # # # # # # # # # # # # # # # # # # # 

def PVM():
    P = ncpol2sdpa.generate_measurements([4]*3, 'P')
    Q = ncpol2sdpa.generate_measurements([4]*3, 'Q')
    F = ncpol2sdpa.generate_measurements([2]*9, 'F')

    return P, Q, F

def PVM_substitutions(P,Q,F):
    substitutions = ncpol2sdpa.projective_measurement_constraints((P,Q,F))
    return substitutions


# # # # # # # # # # # # # # # # # # # # # # # # 
# Objective function
# # # # # # # # # # # # # # # # # # # # # # # # 

def povm_outcome(x,y,c):
    #x,y are 0,1,2
    "On inputs x,y to A and B, and output c by C"
    "finds the pairs (a,a') and (b,b') of outputs by A and B"
    "that makes them win the game"
    if x==0:
        if c==0:
            y_out = (0,1)
        else:
            y_out = (2,3)
    elif x==1:
        if c==0:
            y_out = (0,2)
        else:
            y_out = (1,3)
    elif x==2:
        if c==0:
            y_out = (1,2)
        else:
            y_out = (0,3)
    if y==0:
        if c==0:
            x_out = (0,1)
        else:
            x_out = (2,3)
    elif y==1:
        if c==0:
            x_out = (0,2)
        else:
            x_out = (1,3)
    elif y==2:
        if c==0:
            x_out = (0,3)
        else:
            x_out = (1,2)
    return x_out, y_out

def Pa(P, x, x_out):
    "On input x, finds outcome a=x_out s.t. c=a_y"
    if x_out == 3:
        return 1 - (P[x][0] + P[x][1] + P[x][2])
    else: 
        return P[x][x_out]

def Qb(Q, y, y_out):
    "On input y, finds outcome b=y_out s.t. c=b_x"
    if y_out == 3:
        return 1 - (Q[y][0] + Q[y][1] + Q[y][2])
    else: 
        return Q[y][y_out]


def pi_xy(P, Q, F, x, y):
    # x,y are 0,1,2
    # Translate (x,y) to a number in [9]
    "Creates the operator Pi^xy that outputs a winning c=a_y=b_x"
    k = 1*y+3*x
    (x_out_zero, y_out_zero) = povm_outcome(x,y,0)
    (x_out_one, y_out_one) = povm_outcome(x,y,1)

    obj = F[k][0] * (Qb(Q, y, y_out_zero[0]) + Qb(Q, y, y_out_zero[1]))
    obj += (1 - F[k][0]) * (Pa(P, x, x_out_one[0]) + Pa(P, x, x_out_one[1]))
    obj -= (Pa(P, x, x_out_one[0]) + Pa(P, x, x_out_one[1])) * (Qb(Q, y, y_out_zero[0]) + Qb(Q, y, y_out_zero[1]))

    return obj

def objective_function(P,Q,F):
    obj_function = 0
    for x in range(3):
        for y in range(3):
            obj_function -= pi_xy(P,Q,F,x,y)
    return obj_function


# # # # # # # # # # # # # # # # # # # # # # # # 
#   Constrained inequalities
# # # # # # # # # # # # # # # # # # # # # # # # 

'Constraint of winning probability Alice and Bob'
def piAB_xy(P,Q,x,y):
    # x,y are 0,1,2
    # Translate x,y to a number in [9]
    "Creates the operator Pi^xy_AB that outputs a winning a_y=b_x"
    k = 1*y+3*x
    (x_out_zero, y_out_zero) = povm_outcome(x,y,0)
    (x_out_one, y_out_one) = povm_outcome(x,y,1)

    obj = (Pa(P,x,x_out_zero[0]) + Pa(P,x,x_out_zero[1])) * (Qb(Q,y,y_out_zero[0]) + Qb(Q,y,y_out_zero[1]))
    obj += (Pa(P,x,x_out_one[0]) + Pa(P,x,x_out_one[1])) * (Qb(Q,y,y_out_one[0]) + Qb(Q,y,y_out_one[1]))

    return obj/9

def ineq_AB(P,Q,score):
    constraint = -score
    for x in range(3):
        for y in range(3):
            constraint += piAB_xy(P,Q,x,y)
    inequalities = [constraint]
    return inequalities


'Constraint of winning probability Alice and Charlie'
def piAC_xy(P,F,x,y):
    # x,y are 0,1,2
    # Translate x,y to a number in [9]
    "Creates the operator Pi^xy_AC that outputs a winning c=a_y"
    k = 1*y+3*x
    (x_out_zero, y_out_zero) = povm_outcome(x,y,0)
    (x_out_one, y_out_one) = povm_outcome(x,y,1)

    obj = F[k][0] * (Pa(P,x,x_out_zero[0]) + Pa(P,x,x_out_zero[1]))
    obj += (1 - F[k][0]) * (Pa(P,x,x_out_one[0]) + Pa(P,x,x_out_one[1]))

    return obj/9

def ineq_AC(P,F,eps):
    constraint = eps - 1
    for x in range(3):
        for y in range(3):
            constraint += piAC_xy(P,F,x,y)
    inequalities = [constraint]
    return inequalities


'Constraint of winning probability Bob and Charlie'
def piBC_xy(Q,F,x,y):
    # x,y are 0,1,2
    # Translate x,y to a number in [9]
    "Creates the operator Pi^xy_BC that outputs a winning c=b_x"
    k = 1*y+3*x
    (x_out_zero, y_out_zero) = povm_outcome(x,y,0)
    (x_out_one, y_out_one) = povm_outcome(x,y,1)

    obj = F[k][0] * (Qb(Q,y,y_out_zero[0]) + Qb(Q,y,y_out_zero[1]))
    obj += (1 - F[k][0]) * (Qb(Q,y,y_out_one[0]) + Qb(Q,y,y_out_one[1]))

    return obj/9

def ineq_BC(Q,F,eps):
    constraint = eps - 1
    for x in range(3):
        for y in range(3):
            constraint += piBC_xy(Q,F,x,y)
    inequalities = [constraint]
    return inequalities


# # # # # # # # # # # # # # # # # # # # # # # # 
#   Extra monomials for NPA relaxation
# # # # # # # # # # # # # # # # # # # # # # # # 

def extra_deg3(P,Q,F):
    P_flat = flatten(P)
    Q_flat = flatten(Q)
    F_flat = flatten(F)

    # Adds all PQF operators. 
    # This is too many in general, later we sift out every second, or third, or fourth element as appropriate, depending on your RAM
    extramonomials = [Pa*Qb*Fc for Pa in P_flat for Qb in Q_flat for Fc in F_flat]
    return extramonomials


# # # # # # # # # # # # # # # # # # # # # # # # 
#   We get the relaxation on the desired level
# # # # # # # # # # # # # # # # # # # # # # # # 

def relaxation(level, extramonomials, 
            score=None, eps_AC=None, eps_BC=None, 
            verbose=0, save=False):
    if save:
        dir = 'data_for_level_%s' %level
        os.mkdir(dir)
    
    '''The variables'''
    variables = PVM()
    P = variables[0]
    Q = variables[1]
    F = variables[2]
    variables = ncpol2sdpa.flatten([P,Q,F])

    '''The substitions'''
    substitutions = PVM_substitutions(P,Q,F)

    '''The objective function'''
    objective = objective_function(P,Q,F)

    '''The constrained inequalities'''
    inequalities = []
    if score is not None:
        inequality_AB = ineq_AB(P,Q,score)
        inequalities += inequality_AB
    if eps_AC is not None:
        inequality_AC = ineq_AC(P,F,eps_AC)
        inequalities += inequality_AC
    if eps_BC is not None:
        inequality_BC = ineq_BC(Q,F,eps_BC)
        inequalities += inequality_BC

    '''Getting the relaxation'''
    sdpRelaxation = SdpRelaxation(variables=variables, verbose = verbose)
    if extramonomials:
        extramonomials = extra_deg3(P,Q,F)
        
        # Here we sift out every third PQF operator, else it's too many for our 256GB of RAM. Change as appropriate
        extramonomials = [extramonomials[i] for i in range(len(extramonomials)) if i%3==0]
        sdpRelaxation.get_relaxation(level,     
                                    objective=objective, 
                                    substitutions=substitutions, 
                                    inequalities = inequalities,
                                    extramonomials=extramonomials)
    elif not extramonomials:
        sdpRelaxation.get_relaxation(level, 
                                     objective=objective, 
                                     substitutions=substitutions,
                                     inequalities=inequalities)
    sdpRelaxation.solve()


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
        primal_string = 'The primal objective value is %s. This is an upper bound on the MSG quantum value.' %(-primal_val)
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


# %%

# # # # # # # # # # # # # # # # # # # # # # # # 
# Cons data for MSG
# # # # # # # # # # # # # # # # # # # # # # # # 

# To achieve the best approximation to the 3 party MSG quantum value, set score = 8/9, eps_AC = eps_BC = 1/9

ABC_win = []
AB_win = []
o3 = 8/9
o2 = 1
scores = np.linspace(o3, o2, 20)

for score in scores:
    out = relaxation(level=2, score=score, verbose=0, extramonomials=False)
    ABC_win.append(out[0])
    AB_win.append(out[1])


import matplotlib.pyplot as plt

np.savetxt('MSG__Data/Pr_ABwin', AB_win)
np.savetxt('MSG__Data/Pr_ABCwin', ABC_win)

plt.scatter(AB_win, ABC_win)
plt.savefig('MSG__Data/Pr_ABC_constrained')


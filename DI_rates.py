# %%
import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy
import matplotlib.pyplot as plt
# %%

'''
Based on the code from PBrown from BFF23: DI lower bounds on the conditional von Neumann entropy
'''

# %%

# # # # # # # # # # # # # #
# Get operators
# # # # # # # # # # # # # #

def get_operators(game):
    
    if game == 'CHSH_mod':
        A = [2]*3
        B = [2]*2
        C = 2
    elif game == 'MSG':
        A = [4]*3
        B = [4]*3
        C = 2
    P = ncp.generate_measurements(A, 'P')
    Q = ncp.generate_measurements(B, 'Q')
    Z = ncp.generate_operators('Z', C, hermitian=0)

    return P, Q, Z



# # # # # # # # # # # # # #
# Get objective function
# # # # # # # # # # # # # #

def objective_CHSH_mod(ti, P, Z):
    """
    Returns the objective function for the modified CHSH game for DIQKD on Alice input of 2
    """

    obj = 0
    F = [P[2][0], 1-P[2][0]]
    for a in range(len(F)):
        obj += F[a] * (Z[a] + Dagger(Z[a]) + (1-ti)*Dagger(Z[a])*Z[a]) + ti*Z[a]*Dagger(Z[a])

   
        
    return obj
    

def objective_MSG_small(ti, P, Z, x=0, y=0):
    """
    Returns the objective function for the Magic Square Game on pair of inputs x,y.
    """
    obj = 0

    def Psxy(P,x,y):
        ''' Gets Alice's projections on input of x,y and output s'''
        projections = [[P[i][0] + P[i][1], P[i][0] + P[i][2], P[i][0] + 1-sum(P[i])] for i in range(3)]

        return [projections[x][y], 1-projections[x][y]]
    
    F = Psxy(P,x,y)

    for s in range(2):
        obj += (F[s] * (Z[s] + Dagger(Z[s]) + (1-ti)*Dagger(Z[s])*Z[s]) + \
            ti*Z[s]*Dagger(Z[s]))
        
    return obj


def objective(ti, P, Z, game, x=0, y=0):
    if game == 'CHSH_mod':
        return objective_CHSH_mod(ti, P, Z)
    elif game == 'MSG':
        return objective_MSG_small(ti, P, Z, x, y)



# # # # # # # # # # # # # #
# Get constraints from a Bell inequality
# # # # # # # # # # # # # #

def score_constraints_CHSH_mod(score, P, Q):
    """
    Returns modified CHSH score constraint
    """
    obj = (P[0][0] * Q[0][0] + (1-P[0][0]) * (1-Q[0][0]) + \
        P[0][0] * Q[1][0] + (1-P[0][0]) * (1-Q[1][0]) + \
        P[1][0] * Q[0][0] + (1-P[1][0]) * (1-Q[0][0]) + \
        P[1][0] * (1-Q[1][0]) + (1-P[1][0]) * Q[1][0])/6

    obj += (P[2][0] * Q[0][0] + (1 - P[2][0]) * (1 - Q[0][0]) + 1)/6

    return [obj - score]

def score_constraints_MSG(score, P, Q):
    """
    Returns MSG score constraint
    """
    def piAB_xy(P,Q,x,y):
        '''
        This function creates the operators Pi^xy that output a winning a_y=b_x
        '''
        def povm_outcome(x,y,c):
            "x,y are 0,1,2"
            "POVM labels are ordered as follows:"
            "A: P011,P101,P110,P000=1-..."
            "B: Q001,Q010,Q100,Q111=1-..."
            if x==0:
                if c==0:
                    b_out = (0,1)
                else:
                    b_out = (2,3)
            elif x==1:
                if c==0:
                    b_out = (0,2)
                else:
                    b_out = (1,3)
            elif x==2:
                if c==0:
                    b_out = (1,2)
                else:
                    b_out = (0,3)
            if y==0:
                if c==0:
                    a_out = (0,1)
                else:
                    a_out = (2,3)
            elif y==1:
                if c==0:
                    a_out = (0,2)
                else:
                    a_out = (1,3)
            elif y==2:
                if c==0:
                    a_out = (0,3)
                else:
                    a_out = (1,2)
            return a_out, b_out
        '''
        On inputs x,y to A and B, and output c by C
        brute force the pairs (a,a') and (b,b') of outputs by A and B
        that makes them win the MS game.
        '''
        def Pa(P, x, a_out):
            if a_out == 3:
                return 1 - (P[x][0] + P[x][1] + P[x][2])
            else: 
                return P[x][a_out]
        '''On input x, finds outcome a=a_out s.t. c=a_out'''
        def Qb(Q, y, b_out):
            if b_out == 3:
                return 1 - (Q[y][0] + Q[y][1] + Q[y][2])
            else: 
                return Q[y][b_out]
        '''On input y, finds outcome b=b_out s.t. c=b_out'''
        
        (a_out_zero, b_out_zero) = povm_outcome(x,y,0)
        (a_out_one, b_out_one) = povm_outcome(x,y,1)

        obj = (Pa(P,x,a_out_zero[0]) + Pa(P,x,a_out_zero[1])) * (Qb(Q,y,b_out_zero[0]) + Qb(Q,y,b_out_zero[1]))
        obj += (Pa(P,x,a_out_one[0]) + Pa(P,x,a_out_one[1])) * (Qb(Q,y,b_out_one[0]) + Qb(Q,y,b_out_one[1]))

        return obj

    msg_expr = 0
    for x in range(3):
        for y in range(3):    
            msg_expr += piAB_xy(P,Q,x,y)
    
    return [msg_expr/9 - score]


def score_constraints(score, P, Q, game):
    if game == 'CHSH_mod':
        return score_constraints_CHSH_mod(score, P, Q)
    elif game == 'MSG':
        return score_constraints_MSG(score, P, Q)



# # # # # # # # # # # # # #
# Get substitutions for NPA relaxation
# # # # # # # # # # # # # #

def get_subs(P,Q,Z):
    """
    Returns any substitution rules to use with ncpol2sdpa. E.g. projections and
    commutation relations.
    """
    subs = {}
    # Get Alice and Bob's projective measurement constraints
    subs.update(ncp.projective_measurement_constraints(P,Q))

    # Finally we note that Alice and Bob's operators should All commute with Eve's ops
    for a in ncp.flatten([P,Q]):
        for z in Z:
            subs.update({z*a : a*z, Dagger(z)*a : a*Dagger(z)})

    return subs



# # # # # # # # # # # # # #
# Get extramonomials for NPA relaxation
# # # # # # # # # # # # # #

def get_extra_monomials(P,Q,Z, game,x=0,y=0):
    """
    Returns additional monomials to add to sdp relaxation.
    """
    monos = []
    
    if game =='CHSH_mod':
        # Add all PQZ
        ZZ = Z + [Dagger(z) for z in Z]
        Pflat = ncp.flatten(P)
        Qflat = ncp.flatten(Q)
        for a in Pflat:
            for b in Qflat:
                for z in ZZ:
                    monos += [a*b*z]

        # Add PZZ* monos appearing in objective function
        for z in Z:
            monos += [P[0][0]*Dagger(z)*z]
    
    elif game == 'MSG':
        # Add PQZ for MSG game corresponding to P[x]
        ZZ = Z + [Dagger(z) for z in Z]
        Qflat = ncp.flatten(Q)
        for z in ZZ:
            for q in Qflat:
                for p in P[x]:
                    monos += [p*q*z]

        # Add PZZ* monos appearing in objective function
        for z in Z:
            for a in range(len(P[x])):
                monos += [P[x][a]*Dagger(z)*z]

    return monos

# # # # # # # # # # # # # #
# Setting up the Gauss Radau Quadrature
# # # # # # # # # # # # # #

def generate_quadrature(m):
    t, w = chaospy.quad_gauss_radau(m, chaospy.Uniform(0, 1), 1)
    t = t[0]
    return t, w

def compute_entropy(SDP, P, Z, T, W, KEEP_M, game, VERBOSE):

    ck = 0.0        # kth coefficient
    ent = 0.0        # lower bound on H(A|X=0,E)

    # Best to keep the last step in optimization unless running into numerical problems
    if KEEP_M:
        num_opt = len(T)
    else:
        num_opt = len(T) - 1

    for k in range(num_opt):
        ck = W[k]/(T[k] * log(2))

        # Get the k-th objective function
        new_objective = objective(T[k], P, Z, game)

        SDP.set_objective(new_objective)
        SDP.solve('mosek')

        if SDP.status == 'optimal':
            # 1 contributes to the constant term
            ent += ck * (1 + SDP.dual)
        else:
            # If we didn't solve the SDP well enough then just bound the entropy
            # trivially
            ent = 0
            if VERBOSE:
                print('Bad solve: ', k, SDP.status)
            break

    return ent



# # # # # # # # # # # # # #
# This computes the von Neumann entropy of the input game
# # # # # # # # # # # # # #

def vn_entropy_bound(level, M, KEEP_M, score, verbose=0, extramons=True, game='CHSH'):
    # Get operators first
    P,Q,Z = get_operators(game)

    # Get the objective function (changed later in compute_entropy)
    obj = objective(1, P, Z, game)
    
    subs = get_subs(P,Q,Z) # Projections and commuting substitutions
    moment_ineqs = [] # We only need score constraints, but there might be more
    moment_eqs = [] # Not needed here
    op_eqs = [] # Operator equalities
    op_ineqs = [] # Operator inequalities (e.g. Id-P00 >= 0)

    if extramons:
        extramonomials = get_extra_monomials(P,Q,Z,game)
    else: 
        extramonomials = []
    
    # Creating a test score ---without this you get an index out of range error, for some reason i do not understand
    if game == 'CHSH_mod':
        test_score = 0.85
    elif game == 'MSG':
        test_score = 0.995
    score_cons = score_constraints(test_score, P, Q, game)

    # Creating the sdps
    ops = ncp.flatten([P,Q,Z])
    sdp = ncp.SdpRelaxation(ops, verbose=verbose, normalized=True, parallel=0)
    sdp.get_relaxation(objective = obj,
                        level = level,
                        equalities = op_eqs[:],
                        inequalities = op_ineqs[:],
                        momentinequalities = moment_ineqs[:] + score_cons[:],
                        momentequalities = moment_eqs[:],
                        substitutions = subs,
                        extramonomials = extramonomials)

    
    # Creating the quadrature nodes
    T, W = generate_quadrature(M)
    
    # Evaluates for single score instead of a range (eg. for asymptotic keyrates for a particular qber)

    # Modify the game score
    score_cons = score_constraints(score, P, Q, game)
    sdp.process_constraints(equalities = op_eqs[:],
                            inequalities = op_ineqs[:],
                            momentequalities = moment_eqs[:],
                            momentinequalities = moment_ineqs[:] + score_cons[:])

    # Get the resulting entropy bound
    ent = compute_entropy(sdp, P, Z, T, W, KEEP_M, game, verbose)

    return ent
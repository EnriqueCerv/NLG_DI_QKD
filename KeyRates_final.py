# %%
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

# %%

'''
This file calculates the finite keyrates in Section IV of 
Device independent security of quantum key distribution from arbitrary monogamy-of-entanglement games
'''


# %%
if not os.path.isdir('KeyRate_Data'):
    os.mkdir('Keyrate_Data')

# %% 

# Keys
# x0 = varepsilon_a
# x1 = bstar
# x2 = varepsilon_PA
# x3 = varepsilon_corr
# x4 = varepsilon_s
# x5 = varepsilon_s^p
# x6 = varepsilon_s^pp
# x7 = varepsilon_s^ppp
# x8 = varepsilon_s^pppp
# x9 = alpha


# # # # # # # # # # # # # #
# The soundness constraint
# # # # # # # # # # # # # #
def varepsilon_sound(soundness):
    return lambda x : soundness - (3*x[3] + max([ (x[2]+2*x[4]), 2*(max([x[0],x[4]]))]))


# # # # # # # # # # # # # #
# The length of error correction
# # # # # # # # # # # # # #
def bin_entropy(p):
    if p == 0 or p == 1:
        return 0
    return - p*np.log2(p) - (1-p)*np.log2(1-p)

def qber_CHSH(q):
    o2 = (2+np.sqrt(2))/4
    return q
    return q*(2*o2 -1)

def qber_MSG(q):
    return 2*q*(1-q)

def qber(q, game):
    if game == 'MSG':
        return qber_MSG(q)
    elif game == 'CHSH_mod':
        return qber_CHSH(q)

def leak_EC_approx(n,q,game):
    return 1.1*n*bin_entropy(qber(q,game))


# # # # # # # # # # # # # #
# The parameter estimation constraints via cumulative binomial 
# # # # # # # # # # # # # #
def e_rel_entropy(q,p):
    return q * np.log(q/p) + (1-q) * np.log((1-q)/(1-p))

def varepsilon_PE(n,p,k, completeness):
    if k/n - p < 0:
        sign = -1
    elif k/n - p > 0:
        sign = +1
    else:
        sign = 0
    argument = sign * np.sqrt(2 * n * e_rel_entropy(k/n, p))

    # return norm.cdf(argument)
    return completeness/2 - norm.cdf(argument)


# # # # # # # # # # # # # #
# The parameter estimation constraints via Chernoff
# # # # # # # # # # # # # #
def varepsilon_PE_Chernoff(n, gamma, o2, q, dtol, game, completeness):
    return completeness/2 - np.exp(-n * gamma * (dtol**2) / (2*(1 - (o2-qber(q, game))) + dtol))


# # # # # # # # # # # # # #
# The min-tradeoff function
# # # # # # # # # # # # # #
def h(x, game):
    '''This function is an upper bound on Pr[ABC win] given constraints on Pr[AB win].
    basically it is formed by joining the end of the steps from each value of Pr[ABC win]
    calculated in the NPA hierarchy'''
    '''This is needed for a better min tradeoff function'''

    if game == 'CHSH':
        constraint = np.load('CHSH__Data/Pr_ABwin.npy')
        abc_win = np.load('CHSH__Data/Pr_ABCwin.npy')  # Loads the values of pr[abc win] given constrsints for CHSH, produced from chsh_3.py
    elif game == 'CHSH_mod':
        constraint = np.loadtxt('CHSH_mod__Data/Pr_ABwin')
        abc_win = np.loadtxt('CHSH_mod__Data/Pr_ABCwin')  # Loads the values of pr[abc win] given constrsints for CHSH_mod, produced from chsh_mod.py
    elif game == 'MSG':
        constraint = np.load('MSG__Data/Pr_ABwin.npy')
        abc_win = np.load('MSG__Data/Pr_ABCwin.npy')  # Loads the values of pr[abc win] given constrsints for MSG, produced from msg_AB_constraints.py

    for i in range(len(constraint)-1):
        if x >= constraint[i] and x < constraint[i+1]:
            if i == 0:
                grad = 0
                h_val = abc_win[0]
                return grad, h_val

            rise = abc_win[i] - abc_win[i-1]
            run = constraint[i] - constraint[i-1]        
            grad = rise / run # The gradient between points (x_(k+1), y_k) to (x_(k+2), y_(k+1)) is parallel to the line between (x_k, y_k) and (x_(k+1), y_(k+1))
            c = abc_win[i-1] - grad*constraint[i] # Straight line eq is y = m*x + c, here we find the c
            h_val = grad*x + c
            return grad, h_val
    if x >= constraint[i+1]: # This takes the last segment which the four loop could not manage
        rise = abc_win[i] - abc_win[i-1]
        run = constraint[i] - constraint[i-1]        
        grad = rise / run # The gradient between points (x_(k+1), y_k) to (x_(k+2), y_(k+1)) is parallel to the line between (x_k, y_k) and (x_(k+1), y_(k+1))
        c = abc_win[i-1] - grad*constraint[i] # Straight line eq is y = m*x + c, here we find the c
        h_val = grad*x + c
        return grad, h_val

def g_cons(b, bstar, game):
    h_grad, h_val = h(bstar, game)
    grad = (1-h_grad) / (np.log(2)*(1-bstar+h_val))
    c0 = (h_grad-1) * bstar / (np.log(2)*(1-bstar+h_val))
    c1 = np.log2(1-bstar+h_val)

    return grad*b + c0 - c1

def Maxg(bstar, game):
    return g_cons(1, bstar, game)

def Maxf(bstar, game):
    return Maxg(bstar, game)

def Ming(bstar, game):
    return g_cons(0, bstar, game)

def Minf(bstar, gamma, game):
    return (1-1/gamma) * Maxg(bstar, game) + Ming(bstar, game)/gamma

def MinEpsf(bstar, game):
    return Ming(bstar, game)

def Varf(bstar, gamma, game):
    return (1/gamma) * (Maxg(bstar, game) - Ming(bstar, game))**2   


# # # # # # # # # # # # # #
# The constants for GEAT (last two functions are needed for the O(root(n)) version)
# # # # # # # # # # # # # #
def vartheta_approx(varepsilon):
    return np.log2(2/varepsilon**2)

def V(bstar, gamma, game):
    return np.log2(9) + np.sqrt(2 + Varf(bstar, gamma, game))

def nu():
    return 2*np.log(2) / (1+2*np.log(2))

def exp(bstar, game):
    return (2 * np.log2(2) + Maxf(bstar, game) - MinEpsf(bstar, game))

# %%

# # # # # # # # # # # # # #
# SETTING UP THE OPTIMIZATION
# # # # # # # # # # # # # #

# Keys
# x0 = varepsilon_a
# x1 = bstar
# x2 = varepsilon_PA
# x3 = varepsilon_corr
# x4 = varepsilon_s
# x5 = varepsilon_s^p
# x6 = varepsilon_s^pp
# x7 = varepsilon_s^ppp
# x8 = varepsilon_s^pppp
# x9 = alpha

def lkey_approxEC(n, q, o2, gamma, dtol, game):

    if game == 'MSG':
        SAB = 4
    else:
        SAB = 2

    return lambda x : -(n*g_cons(o2-qber(q,game)-dtol, x[1], game) \
        - np.sqrt(n)* np.sqrt(2*np.log(2)*V(x[1],gamma,game)/nu() *  (vartheta_approx(x[4]) + (2-nu())*np.log2(1/x[0]))) \
        - (((2-nu()) * (nu()**2) * np.log2(1/x[0]) + (nu()**2)*vartheta_approx(x[4])) / (3*(np.log(2)**2)*(V(x[1],gamma, game)**2)*((2*nu()-1)**3))) * (2**((1-nu())/nu() * exp(x[1],game))) * np.log((2**(exp(x[1],game))) + np.exp(2))**3\
        - SAB * n * gamma \
        - np.ceil(np.log2(1/x[3])) \
        - leak_EC_approx(n,q,game)) \
        - 2*np.log2(1/x[2])

def constraints(soundness):
    cons = ({'type':'ineq', 'fun': varepsilon_sound(soundness)}) #the soundness parameter
    return cons

# %%

# # # # # # # # # # # # # #
# Function for list of keyrates
# # # # # # # # # # # # # #

def keyrates(q, game, start=4, stop=16, step=1, sound = 10**-6):
    '''
    This function gets keyrates from desired game and soundness, completeness parameters from 10^start to 10^stop
    '''
    keyrate = []

    # Keys
    # x0 = varepsilon_a
    # x1 = bstar
    # x2 = varepsilon_PA
    # x3 = varepsilon_corr
    # x4 = varepsilon_s
    # x5 = varepsilon_s^p
    # x6 = varepsilon_s^pp
    # x7 = varepsilon_s^ppp
    # x8 = varepsilon_s^pppp
    # x9 = alpha

    if game == 'MSG':
        o2 = 1
        o3 = 8/9
        init_guess = np.array([1-10**-6,o2-10**-6,1-10**-6,1-10**-6,1-10**-6]) 
    elif game == 'CHSH_mod':
        o2 = ((np.sqrt(2)+2)/4)*(2/3) + 1/3
        o3 = 1/2+1/3
        init_guess = np.array([2*10**-4,o2-10**-6,1-10**-6,1-10**-4,1-10**-6])    
    
    bounds = ((10**-10,1-10**-6),(o3,o2),(10**-10,1-10**-6),(10**-10,1-10**-6),(10**-10,1-10**-6))

    for n in range(start, stop+1, step):
        block = 10**n
        gamma = 1/(block**(1/3)) * np.log(1/0.005)
        dtol = 1/(block**(2/3))
        result = minimize(lkey_approxEC(block,q,o2,gamma,dtol,game), x0=init_guess, method='Nelder-Mead', bounds=bounds, constraints=constraints(sound))
        
        keyrate.append(-result.fun/block)
        init_guess = result.x


    return keyrate
            
# %%

# # # # # # # # # # # # # #
# Finite keyrates for MSG
# # # # # # # # # # # # # #

o2 = 1
o3 = 8/9
sound = 10**(-6)
com = 10**(-2)

MSG_rates_q0 = keyrates(10**-4, 'MSG')
MSG_rates_q05 = keyrates(0.005, 'MSG')
MSG_rates_q1 = keyrates(0.01, 'MSG')
MSG_rates_q15 = keyrates(0.015, 'MSG')

# %%

# # # # # # # # # # # # # #
# Finite keyrates for CHSH_mod
# # # # # # # # # # # # # #

o2 = ((np.sqrt(2)+2)/4)*(2/3) + 1/3
o3 = 1/2+1/3
sound = 10**(-6)
com = 10**(-2)

CHSH_mod_rates_q0 = keyrates(10**-4, 'CHSH_mod')
CHSH_mod_rates_q1 = keyrates(0.01, 'CHSH_mod')
CHSH_mod_rates_q2 = keyrates(0.02, 'CHSH_mod')
CHSH_mod_rates_q3 = keyrates(0.03, 'CHSH_mod')
# %%

# # # # # # # # # # # # # #
# Graphing the finite keyrates
# # # # # # # # # # # # # #

rounds = [10**block for block in range(4,16+1,1)]

o2 = 1
o3 = 8/9
plt.figure()

plt.scatter(rounds, np.array(MSG_rates_q0), color='black')
plt.plot(rounds, np.array(MSG_rates_q0), color='black')
plt.plot(rounds, [g_cons(o2,1,'MSG')]*len(rounds), color='black', linestyle='dashed')

plt.scatter(rounds, np.array(MSG_rates_q05), color='red')
plt.plot(rounds, np.array(MSG_rates_q05), color='red')
plt.plot(rounds, [(g_cons(o2-qber_MSG(0.005),o2-qber_MSG(0.005),'MSG')-1.1*bin_entropy(qber_MSG(0.005)))]*len(rounds), color='red', linestyle='dashed')

plt.scatter(rounds, np.array(MSG_rates_q1), color='green')
plt.plot(rounds, np.array(MSG_rates_q1), color='green')
plt.plot(rounds, [(g_cons(o2-qber_MSG(0.01),o2-qber_MSG(0.01),'MSG')-1.1*bin_entropy(qber_MSG(0.01)))]*len(rounds), color='green', linestyle='dashed')

plt.scatter(rounds, np.array(MSG_rates_q15), color='blue')
plt.plot(rounds, np.array(MSG_rates_q15), color='blue')
plt.plot(rounds, [(g_cons(o2-qber_MSG(0.015),o2-qber_MSG(0.015),'MSG')-1.1*bin_entropy(qber_MSG(0.015)))]*len(rounds), color='blue', linestyle='dashed')

plt.xscale('log')
plt.xlabel('Block size')
plt.ylabel('Keyrate')
plt.ylim(bottom=0)
plt.ylim(top=1)
plt.xlim(left=10**3)
plt.savefig('Keyrate_Data/MSG_keyrates.png')
plt.show()
plt.close()


o2 = ((np.sqrt(2)+2)/4)*(2/3) + 1/3
o3 = 1/2+1/3
plt.figure()
plt.scatter(rounds, np.array(CHSH_mod_rates_q0)/2, color='black')
plt.plot(rounds, np.array(CHSH_mod_rates_q0)/2, color='black')
plt.plot(rounds, [g_cons(o2,o2,'CHSH_mod')/2]*len(rounds), color='black', linestyle='dashed')

plt.scatter(rounds, np.array(CHSH_mod_rates_q1)/2, color='red')
plt.plot(rounds, np.array(CHSH_mod_rates_q1)/2, color='red')
plt.plot(rounds, [(g_cons(o2-qber_CHSH(0.01),o2-qber_CHSH(0.01),'CHSH_mod')-1.1*bin_entropy(qber_CHSH(0.01)))/2]*len(rounds), color='red', linestyle='dashed')

plt.scatter(rounds, np.array(CHSH_mod_rates_q2)/2, color='green')
plt.plot(rounds, np.array(CHSH_mod_rates_q2)/2, color='green')
plt.plot(rounds, [(g_cons(o2-qber_CHSH(0.02),o2-qber_CHSH(0.02),'CHSH_mod')-1.1*bin_entropy(qber_CHSH(0.02)))/2]*len(rounds), color='green', linestyle='dashed')

plt.scatter(rounds, np.array(CHSH_mod_rates_q3)/2, color='blue')
plt.plot(rounds, np.array(CHSH_mod_rates_q3)/2, color='blue')
plt.plot(rounds, [(g_cons(o2-qber_CHSH(0.03),o2-qber_CHSH(0.03),'CHSH_mod')-1.1*bin_entropy(qber_CHSH(0.03)))/2]*len(rounds), color='blue', linestyle='dashed')

plt.xscale('log')
plt.xlabel('Block size')
plt.ylabel('Keyrate')
plt.ylim(bottom=0)
plt.ylim(top=0.35)
plt.xlim(left=10**3)
plt.savefig('Keyrate_Data/CHSH_mod_keyrates.png')
plt.show()
plt.close()
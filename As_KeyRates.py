# %%
from DI_rates import vn_entropy_bound
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
import os

# %%

'''
This file calculates the asypmtotic keyrates from Section IV of 
Device independent security of quantum key distribution from arbitrary monogamy-of-entanglement games
based on the numerical techniqes of BFF23
'''


# %%

if not os.path.isdir('KeyRate_Data'):
    os.mkdir('Keyrate_Data')

def bin_entropy(p):
    if p == 0 or p == 1:
        return 0
    return - p*np.log2(p) - (1-p)*np.log2(1-p)


# # # # # # # # # # # # # #
# Function computing H(A|E), H(A|B) for a range of scores
# # # # # # # # # # # # # #

def asymptotic_rate(game, level, n_points, extramonomials):
    t0 = time.time()

    if game == 'CHSH_mod': 
        o2 = ((2+np.sqrt(2))/4)*(2/3) + 1/3
        o3 = 1/2 + 1/3
    elif game == 'MSG':
        o2 = 1
        o3 = 8/9
    
    vn_AB = []
    vn_AE = []
    scores = np.linspace(o3, o2-10**-4, n_points).tolist()

    for score in scores:
        # The H(A|BX) term
        qber = o2 - score
        vn_AB += [bin_entropy(qber)]

        # The H(A|EXY)/H(A|EX)term
        output = vn_entropy_bound(level=level, M=8, KEEP_M=1, score=score, game=game, extramons=extramonomials)
        vn_AE += [output]
        
        print('Done with score ', score, 'with entropy ', output[1])
        t1 = time.time()
        print('That took ', (t1-t0)/60, 'min')
        t0 = t1

    return scores, vn_AB, vn_AE


# %%

# # # # # # # # # # # # # #
# Asymptotic keyrates for modified CHSH game
# # # # # # # # # # # # # #

scores, vn_AB, vn_AE = asymptotic_rate('CHSH_mod', 2, 15, True)

np.savetxt('Keyrate_Data/CHSH_mod_vn_scores', scores)
np.savetxt('Keyrate_Data/CHSH_mod_vn_AB_entropies', vn_AB)
np.savetxt('Keyrate_Data/CHSH_mod_vn_AE_entropies', vn_AE)

dw = np.array(vn_AE)-np.array(vn_AB)
plt.plot(scores, vn_AE, label='H(S|X=2;E)')
plt.scatter(scores, vn_AE)
plt.plot(scores, dw, label='DW rate')
plt.scatter(scores, dw)
plt.xlabel('Pr[AB win]')
plt.ylabel('Bits')
plt.ylim(bottom=0)
plt.xlim(left=1/3+1/2)
plt.legend()
plt.savefig('Keyrate_Data/CHSH_As_keyrates')
plt.show()

# %%

# # # # # # # # # # # # # #
# Asymptotic keyrates for MSG
# # # # # # # # # # # # # #

scores, vn_AB, vn_AE = asymptotic_rate('MSG', 2, 15, True)

np.savetxt('Keyrate_Data/MSG_vn_scores', scores)
np.savetxt('Keyrate_Data/MSG_vn_AB_entropies', vn_AB)
np.savetxt('Keyrate_Data/MSG_vn_AE_entropies', vn_AE)

dw = np.array(vn_AE)-np.array(vn_AB)
plt.plot(scores, vn_AE, label='H(S|X=0,Y=0;E)')
plt.scatter(scores, vn_AE)
plt.plot(scores, dw, label='DW rate')
plt.scatter(scores, dw)
plt.xlabel('Pr[AB win]')
plt.ylabel('Bits')
plt.ylim(bottom=0)
plt.xlim(left=8/9)
plt.legend()
plt.savefig('Keyrate_Data/MSG_As_keyrates')
plt.show()
# %%

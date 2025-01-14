# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:01:24 2023

@author: burak
"""

import numpy as np

def randomize(eps, num_classes, true_val):  
    """

    Parameters
    ----------
    eps : float
        Privacy parameter, epsilon. Smaller epsilon means better privacy. range: (0,inf).
    num_classes : int
        Number of possible classes of the classifier.
    true_val : int
        The index of true class. range: {0,1,...,num_classes}

    Returns
    -------
    chosen_ind : int
        Index of the randomized class.

    """
    probs = np.ones(num_classes)
    probs[true_val] = np.exp(eps)
    probs = probs/np.sum(probs)
    chosen_ind = np.random.choice(num_classes, p=probs)
    return chosen_ind

def final_eps(eps_local, delta, n):
    """
    Calculating the epsilon value after shuffling. 
    The formula is from Thm 3.1 of https://arxiv.org/pdf/2012.12803.pdf
    This function is intended to be used inside get_local_eps function.

    Parameters
    ----------
    eps_local : float
        epsilon-DP guarantee of local randomizers (per client, before shuffling)
    delta : float
        Final desired delta
    n : integer
        Number of clients

    Returns
    -------
    float
        Desired epsilon after shuffling

    """
    return np.log(1+
                  ((np.exp(eps_local)-1)/(np.exp(eps_local)+1))
                  *(8*np.sqrt(np.exp(eps_local)*np.log(4/delta)/n)
                    +8*np.exp(eps_local)/n)
                  )


def get_local_eps(target_eps, delta, n):
    """
    Binary search algoritm to find epsilon value of local randomizers of each client

    Parameters
    ----------
    target_eps : float
        Desired epsilon after shuffling
    delta : float
        Desired delta after shuffling
    n : int
        Number of clients

    Returns
    -------
    float
        Approximate value of the local epsilon value

    """
    eps_min = target_eps
    eps_max = target_eps*10
    
    while np.abs(eps_max-eps_min)>1e-15 :
        avg = 0.5*(eps_max + eps_min)
        if final_eps(avg, delta, n) > target_eps:
            eps_max = avg
        else:
            eps_min = avg
    return 0.5*(eps_max + eps_min)
    


#print(randomize(1.0, 5, 1))
print(get_local_eps(1.0, 1e-6, 20))

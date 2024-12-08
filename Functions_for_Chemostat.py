import numpy as np
import numpy.ma as ma
import scipy.optimize
import matplotlib.pyplot as plt
import math
import time
import pandas as pd
import seaborn as sns
from matplotlib import ticker, cm
from functools import partial
from scipy.integrate import odeint
from scipy.optimize import fsolve

def growth(a, B, T, h, ec50, **kwargs):
    return B + (T-B)/(1 + (a/ec50)**h)

def chemostat_model(model_state, t, parms):
    g = model_state[0]
    a = model_state[1]
    R = np.array(model_state[2])
    S = np.array(model_state[3])
    if a <= 0:
        a = 1e-9
    yxg = parms[0].get('yxg')
    k = parms[0].get('k')
    ka_R = parms[0].get('ka')
    ka_S = parms[1].get('ka')
    Vmax_R = parms[0].get('Vmax')
    Vmax_S = parms[1].get('Vmax')
    d = parms[2].get('d')
    feed_g = parms[2].get('feed_g')
    feed_a = parms[2].get('feed_a')
    d_e_a = parms[2].get('degradation_a') # extra degradation term to simulate pd dynamics

    # the growth rates
    mu_R = growth(a, **parms[0])
    mu_S = max(growth(a, **parms[1]), 0)
    d_S = min(0, growth(a, **parms[1]))
    
    # the derivatives
    dgdt = feed_g * d - (1/yxg)*(R*(g/(k+g))*mu_R + S*(g/(k+g))*mu_S) - d*g
    dadt = feed_a * d - Vmax_R*R*(a/(a + ka_R)) - Vmax_S*S*(a/(a + ka_S)) - d*a - d_e_a * a
    dRdt = R*(g/(k+g))*mu_R - d*R
    dSdt = S*(g/(k+g))*mu_S + S*d_S - d*S

    return np.array([dgdt, dadt, dRdt, dSdt])

# Function to repeatedly simulate a chemostat with 100 timesteps per hour
def serial_chemostat(model, init, iterations, interval, amount, parms):
    timeadd = 0
    sol = []
    t = np.linspace(0,interval,int(np.round(interval*100)))
    for i in range(0, iterations):
        #set up model
        sol = odeint(chemostat_model, init, t, args=parms)
        #adjust the time
        if i==0:
            totalsol = sol
        else:
            totalsol = np.concatenate((totalsol,sol))
        #reset
        init = sol[-1]+np.array([0,amount,0,0])
        sol = []
    return totalsol

# This functions gives the ratio of the triple at a certain model state
def calc_ratio(model_state):
    return model_state[2]/(model_state[2]+model_state[3])

# These functions calculate the equilibrium ratio of the Triple in the chemostat by first checking high and low ratios, doing 3 iterations and checking if the ratio decreases between the second and third iteration (because the chemostat will not be steady state and in the beginning there are no antibiotics, we take the second and third iteration)
def equilibrium_ratio_serial(feed_g, interval, amount, d, d_e_a, p_R, p_S, sensitivity=0.001, max_iterations=50):
    func_equal_growth = lambda a: (growth(a, **p_R) - growth(a, **p_S))**2
    sol_a_eq = fsolve(func_equal_growth, 0.6)[0] #0.6 is the initial value
    approx_ss_popsize = N_tot(feed_g, d, g_min(d, growth(sol_a_eq, **p_R), **p_R), mu_coex = growth(sol_a_eq, **p_R), **p_R)
    approx_gly = g_min(d, growth(0, **p_R), **p_R)
    ratio_low = 0 + sensitivity
    ratio_high = 1 - sensitivity
    iterations = 0
    p_chemostat = {"d": d, "feed_g": feed_g, "feed_a": 0, "degradation_a": d_e_a}
    init_low = np.array([approx_gly, amount, (1-ratio_high)*approx_ss_popsize, ratio_high*approx_ss_popsize])
    sim_low = serial_chemostat(chemostat_model, init_low, 3, interval, amount, parms=([p_R, p_S, p_chemostat],))
    sim_low_start = sim_low[sim_low[:,1]>(amount-0.0001)]
    if calc_ratio(sim_low_start[-1]) < calc_ratio(sim_low_start[0]):
        return 0
    init_high = np.array([approx_gly, amount, (1-ratio_low)*approx_ss_popsize, ratio_low*approx_ss_popsize])
    sim_high = serial_chemostat(chemostat_model, init_high, 3, interval, amount, parms=([p_R, p_S, p_chemostat],))
    sim_high_start = sim_high[sim_high[:,1]>(amount-0.0001)]    
    if calc_ratio(sim_high_start[-1]) > calc_ratio(sim_high_start[0]):
        return 1
    while abs(ratio_high - ratio_low) > sensitivity:
#        print('iteration', iterations)
        if iterations == max_iterations:
            print("Precision not reached: Difference is", abs(ratio_high - ratio_low))
            break
        ratio_check = ratio_low + 0.5*(ratio_high - ratio_low)
        init_check = np.array([approx_gly, amount, (ratio_check)*approx_ss_popsize, (1-ratio_check)*approx_ss_popsize])
        sim = serial_chemostat(chemostat_model, init_check, 3, interval, amount, parms=([p_R, p_S, p_chemostat],))
        sim_start = sim[sim[:,1]>(amount-0.0001)]
#        print('ratio', ratio_check, 'sim_start', sim_start)
        if calc_ratio(sim_start[-1]) > calc_ratio(sim_start[0]):
            ratio_low = ratio_check
        elif calc_ratio(sim_start[-1]) < calc_ratio(sim_start[0]):
            ratio_high = ratio_check
        iterations += 1
    return ratio_low + (ratio_high-ratio_low)/2

# Calculations for chemostat based on continuous culture

#We find the glycerol concentration under which the strains cannot grow depending on the dilution rate (assuming not antibiotics; so this is based on the growth of the susceptible strain, because that one grows faster without antibiotics).

def g_min(d, mu, k, **kwargs):
    return d*k/(mu-d)

# The boundary between the area where the resistant wins and the coexistence area: We define a function that gives the feed concentration of glycerol depending on the antibiotic feed concentration below which the resistant cannot break down the antibiotic concentration to under sol_a_eq even if the whole population is resistant

def feed_g_R_coex(feed_a, d, a, ka, Vmax, muR, k, yxg, **kwargs):
    return ((1/yxg)*((feed_a - a)*(a + ka)/(a*Vmax))*(muR/(k + d*k/(muR - d)))+1)*(d*k/(muR-d))

# The boundary between the area where the susceptible wins and the coexistence area: We define a function that gives the feed concentration of glycerol depending on the antibiotic feed concentration above which the susceptible is so abundant that it breaks down the antibiotic until under sol_a_eq by itself

def feed_g_coex_S(feed_a, d, a, ka, Vmax, muS, k, yxg, **kwargs):
    return ((1/yxg)*((feed_a - a)*(a + ka)/(a*Vmax))*(muS/(k + d*k/(muS - d)))+1)*(d*k/(muS-d))

# The above two equations seem exactly the same, only the parameters come from the R of the S strain (so the ka and Vmax)... It would be better to have one equation only...

# Then we calculate the total cell density as a function glycerol feed, assuming that there is coexistence (a = sol_a_eq)

def N_tot(feed_g, d, g, mu_coex, k, yxg, **kwargs):
    return (k+g)*(feed_g-g)*d*yxg/(g*mu_coex)

# Then we calculate the density of the resistant strain as a function glucerol and antibiotics feed, assuming that there is coexistence (a = sol_a_eq)

def N_R_coex(feed_a, d, a, N_tot, parS, parR, **kwargs):
    ka_R = parR.get('ka')
    ka_S = parS.get('ka')
    Vmax_R = parR.get('Vmax')
    Vmax_S = parS.get('Vmax')
    return ((feed_a-a)*d - Vmax_S*N_tot*a/(a+ka_S))/(Vmax_R*a/(a+ka_R) - Vmax_S*a/(a+ka_S))

# Calculate the equilibrium ratio and give back:

# -1 => no strain can grow (the glycerol in the feed is too low to achieve the growth rate of d at a=0)
# 0 => susceptible only (if a < sol_a_eq or density is high enough for S alone to break down a until under sol_a_eq)
# 1 => resistant only (if density is so low that R alone does not break down a until under sol_a_eq) number between 0 and 1 => equilibrium ratio of R

def equilibrium_ratio(a, feed_g, d, sol_a_eq, p_S, p_R):
    if feed_g < g_min(d, mu=growth(0, **p_S), **p_S):
        return -1
    elif a < sol_a_eq:
        return 0 
    elif feed_g < feed_g_R_coex(a, d, sol_a_eq, muR=growth(sol_a_eq, **p_R), **p_R):
        return 1
    elif feed_g > feed_g_coex_S(a, d, sol_a_eq, muS=growth(sol_a_eq, **p_S), **p_S):
        return 0
    else:
        N_tot_temp = N_tot(feed_g, d, g_min(d, growth(sol_a_eq, **p_R), **p_R), mu_coex = growth(sol_a_eq, **p_R), **p_R)
        return N_R_coex(a, d, sol_a_eq, N_tot_temp, p_S, p_R)/N_tot_temp
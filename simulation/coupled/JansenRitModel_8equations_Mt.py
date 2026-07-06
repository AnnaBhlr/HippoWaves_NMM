# -*- coding: utf-8 -*-
"""
================================================================================
PROVENANCE
================================================================================
This module is DERIVED from the Jansen-Rit neural mass model implementation by:

    Carlos Coronel  (original file: carlos_coronel_original/JansenRitModel_Coronel-Oliveros.py)
    Created Mon Apr 16 14:57:25 2018

The original is a modified Jansen & Rit (1995) neural mass model with an extra
local inhibitory-interneuron -> excitatory-interneuron connection (scaled by
'beta') and excitatory (pyramidal->pyramidal) long-range coupling. The model has
8 state variables per node (x0-x3, y0-y3): three local populations plus a
delayed long-range excitatory term (x3/y3, scaled by 'ad').

Original references:
[1] Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked
    potential generation in a mathematical model of coupled cortical columns.
    Biological Cybernetics, 73(4), 357-366.
[2] Silberberg, G., & Markram, H. (2007). Disynaptic inhibition between
    neocortical pyramidal cells mediated by Martinotti cells. Neuron, 53(5),
    735-746.
[3] Fino, E., Packer, A. M., & Yuste, R. (2013). The logic of inhibitory
    connectivity in the neocortex. The Neuroscientist, 19(3), 228-237.

The core ODE system (sigmoid `s` and the eight derivatives in `f1`) is
unchanged from Coronel's formulation. Please cite the references above when
using this code.

================================================================================
WHAT WAS CHANGED IN THIS DERIVED VERSION (and why)
================================================================================
Modified by A. Behler for the cortex<->hippocampus traveling-wave simulations.

1. Time-switchable connectivity (M1 / M2).
   `f1(...)` and `Sim(...)` now take TWO coupling matrices, M1 and M2, and two
   switch times, t_switch1 and t_switch2. The active matrix is selected by
   simulation time:  M = M1 if (t < t_switch1 or t > t_switch2) else M2.
   WHY: lets us turn inter-regional coupling on/off (or swap topologies) during
   a single run, e.g. to model the onset of cortex<->hippocampus coupling.
   NOTE: `f1` is now called with physical time `i*dt` (seconds) instead of the
   integer step index `i`, so the t_switch values are interpreted in seconds.

2. Per-node (spatially varying) parameters.
   A, B, a, b, ad and the sigmoid slope r are passed as arguments to `f1` and
   may be either scalars or length-`nnodes` vectors (scalars are broadcast).
   WHY: enables spatial gradients of excitability / synaptic time constants /
   intrinsic frequency across the cortical and hippocampal surface, instead of
   the single global value used in the original.

3. Random initial conditions.
   `Sim` now draws initial conditions from np.random.uniform(-1, 1, ...)
   instead of the fixed deterministic state vector used in the original.

4. Removed from the original `Sim`: the M square/dtype validation, the internal
   `set_seed(seed)` call (seeding is no longer applied inside Sim), and the
   EEG/BOLD signal-generation portions.

5. Default coupling constants changed: alpha 0 -> 0.5, beta 0 -> 0.25.

A line-by-line comparison against carlos_coronel_original/JansenRitModel_Coronel-Oliveros.py
shows the derivative equations themselves are identical.
================================================================================
"""
import numpy as np
from numba import jit,float64, vectorize
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


#Simulation parameters
dt = 1E-3 #Integration step
teq = 60 #Simulation time for stabilizing the system
tmax = 600 #Simulation time
downsamp = 10 #Downsampling to reduce the number of points        
seed = 0 #Random seed

#Node parameters
a = 100 #Velocity constant for EPSPs (1/sec)
ad = 50 #Velocity of constant long-range EPSPs (1/sec)
b = 50 #Velocity constasnt for IPSPs (1/sec)
p = 2 #. input to pyramidal population
sigma = 2 #Input standard deviation

C = 135 #Global synaptic connectivity
C1 = C * 1 #Connectivity between pyramidal pop. and excitatory pop.
C2 = C * 0.8 #Connectivity between excitatory pop. and pyramidal pop.
C3 = C * 0.25 #Connectivity between pyramidal pop. and inhibitory pop.
C4 = C * 0.25 #Connectivity between inhibitory pop. and pyramidal pop.

A = 3.25 #Amplitude of EPSPs
B = 22 #Amplitude of IPSPs

#Both as multiples of C
alpha = 0.5  #Long-range pyramidal-pyramidal coupling
beta = 0.25 #Connectivity between inhibitory pop. and excitatory interneuron pop. (short-range)

#Sigmoid function parameters
e0 = 2.5 #Half of the maximum firing rate
v0 = 6 #V1/2
r0 = 0.56 #Slopes of sigmoid function


@vectorize([float64(float64,float64)],nopython=True)
#Sigmoid function
def s(v,r0):
    return (2 * e0) / (1 + np.exp(r0 * (v0 - v)))


#@jit(nopython=True)
#Jansen & Rit multicolumn model (intra-columnar outputs)
def f1(y, t, M1, M2, t_switch1,t_switch2, p, A, B, a, b, ad,r):
    # Use M1 if t is less than t_switch, else use M2
    M = M1 if t < t_switch1 or t > t_switch2 else M2
    
    x0, x1, x2, x3, y0, y1, y2, y3 = y
    nnodes = M.shape[0]
    noise = np.random.normal(0,sigma,nnodes)
    if isinstance(p, (int, float)):  # Check if p is a scalar
        p = np.ones(nnodes) * p
    if isinstance(A, (int, float)):
        A = np.ones(nnodes) * A
    if isinstance(B, (int, float)):
        B = np.ones(nnodes) * B
    if isinstance(a, (int, float)):
        a = np.ones(nnodes) * a
    if isinstance(b, (int, float)):
        b = np.ones(nnodes) * b
    if isinstance(ad, (int, float)):
        ad = np.ones(nnodes) * ad
    
    x0_dot = y0
    y0_dot = A * a * (s(C2 * x1 - C4 * x2 + C * alpha * x3, r)) - \
             2 * a * y0 - a**2 * x0 
    x1_dot = y1
    y1_dot = A * a * (p + noise + s(C1 * x0 - C * beta * x2, r)) - \
             2 * a * y1 - a**2 * x1
    x2_dot = y2
    y2_dot = B * b * (s(C3 * x0, r)) - \
             2 * b * y2 - b**2 * x2
    x3_dot = y3
    y3_dot = A * ad * (M @ s(C2 * x1 - C4 * x2 + C * alpha * x3, r)) - \
             2 * ad * y3 - ad**2 * x3
    
    
    return (np.vstack((x0_dot, x1_dot, x2_dot, x3_dot, y0_dot, y1_dot, y2_dot, y3_dot)))


#@jit(float64(float64),nopython=True)
#This function is just for setting the random seed
#def set_seed(seed):
#    np.random.seed(seed)
#    return(seed)
@jit(nopython=True)
def set_seed(seed):
    seed = int(seed)  # Convert seed to integer
    np.random.seed(seed)
    return seed


def Sim(M1, M2, t_switch1, t_switch2,  verbose=False):

    global teq, tmax, ttotal, downsamp, seed
     
    # Generate random values 
    ic = np.random.uniform(-1, 1, size=(1, nnodes, 8))
    # Reshape to match your original shape (8, nnodes)
    ic = ic.reshape(8, nnodes)

    ttotal = teq + tmax #Total simulation time
    Nsim = int(ttotal / dt) #Total simulation time points
    Neq = int(teq / dt / downsamp) #Number of points to discard
    Nmax = int(tmax/dt / downsamp) #Number of points of final simulated recordings
    Ntotal = Neq + Nmax #Total number of points of total simulated recordings
   
    #Time vector
    time_vector = np.linspace(0, ttotal, Ntotal)

    row = 8 #Number of variables of the Jansen & Rit model
    col = nnodes #Number of nodes
    y_temp = np.copy(ic) #Temporal vector to update y values
    y = np.zeros((Ntotal, row, col)) #Matrix to store values
    y[0,:,:] = np.copy(ic) #First set of initial conditions
    

    if verbose == True:
        for i in range(1, Nsim):
            y_temp += dt * f1(y_temp, i*dt, M1, M2, t_switch1, t_switch2,p, A, B, a, b, ad, r0)
            #This line is for store values each dws points
            if (i % downsamp) == 0:
                y[i//downsamp,:,:] = y_temp
            if (i % (10 / dt)) == 0:
                print('Elapsed time: %i seconds'%(i * dt)) #this is for impatient people
    else:
        for i in range(1,Nsim):
            y_temp += dt * f1(y_temp, i*dt, M1, M2, t_switch1,t_switch2, p, A, B, a, b, ad,r0)
            #This line is for store values each dws points
            if (i % downsamp) == 0:
                y[i//downsamp,:,:] = y_temp 
       
    return(y, time_vector)


def ParamsNode():
    pardict={}
    for var in ('a','b','A','B','r0',
                'r1','r2','e0','v0','C','C1','C2','C3',
                'C4','alpha','beta','p','sigma'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsNet():
    pardict={}
    for var in ('nnodes','mean_speed','std_speed'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsSim():
    pardict={}
    for var in ('tmax','teq','dt','downsamp'):
        pardict[var]=eval(var)
        
    return pardict



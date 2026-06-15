# -*- coding: utf-8 -*-
# =============================================================================
# PROVENANCE
# This model is DERIVED from the original Jansen-Rit implementation by Carlos
# Coronel (kept verbatim in ../../carlos_coronel_original/). The eight-equation
# ODE system (sigmoid `s` and the derivatives in `f1`) is unchanged from the
# original. What was changed here for the hippocampus surface model:
#   - the node parameters A, B, a, b, ad are passed as arguments to f1 and may
#     be per-node vectors (enabling spatial gradients across the surface),
#     instead of the single global scalars used in the original;
#   - driven by a single (static) coupling matrix M.
# (The coupled cortex<->hippocampus variant additionally adds time-switchable
# connectivity; see ../coupled/JansenRitModel_8equations_Mt.py.)
# Please cite Carlos Coronel's work and the Jansen & Rit references below.
# =============================================================================
"""
Created on Mon Apr 16 14:57:25 2018

@author: Carlos Coronel

Modified version of the Jansen and Rit Neural Mass Model [1]. We included an extra
local connection from inhibitory interneurons to excitatory interneurons [2,3], scaled by
a connectivity constant 'beta'. Long-range connections are only excitatory (pyramidal to
pyramidal). The script runs the model for generate EEG-like and BOLD-like signals.

[1] Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked 
potential generation in a mathematical model of coupled cortical columns. 
Biological cybernetics, 73(4), 357-366.

[2] Silberberg, G., & Markram, H. (2007). Disynaptic inhibition between neocortical
pyramidal cells mediated by Martinotti cells. Neuron, 53(5), 735-746.

[3] Fino, E., Packer, A. M., & Yuste, R. (2013). The logic of inhibitory 
connectivity in the neocortex. The Neuroscientist, 19(3), 228-237. 
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
r = 0.56 #Slopes of sigmoid function


@vectorize([float64(float64,float64)],nopython=True)
#Sigmoid function
def s(v,r0):
    return (2 * e0) / (1 + np.exp(r0 * (v0 - v)))


#@jit(nopython=True)
#Jansen & Rit multicolumn model (intra-columnar outputs)
def f1(y,t,M,p,A, B, a, b, ad):
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
        
    #check that
        # NOTE: for local couplings
        # 0: pyramidal cells
        # 1: excitatory interneurons
        # 2: inhibitory interneurons
        # 0 -> 1,
        # 0 -> 2,
        # 1 -> 0,
        # 2 -> 0,
    

    
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


def Sim(M, verbose=False):
    
    """
    Run a network simulation with the current parameter values.
    
    Note that the time unit in this model is seconds.

    Parameters
    ----------
    verbose : Boolean, optional
        If True, some intermediate messages are shown.
        The default is False.

    Raises
    ------
    ValueError
        An error raises if the dimensions of M and the number of nodes
        do not match.

    Returns
    -------
    y : ndarray
        Time trajectory for the six variables of each node.
    time_vector : numpy array (vector)
        Values of time.
    z : numpy array (matrix)
        long-range inputs over time to each node.

    """
    #global teq,tmax,ttotal,downsamp,M,D,seed
    global teq, tmax, ttotal, downsamp, seed  # removed M, D from the list
    
    #Structural connectivity

    # Check if M is square
    if M.shape[0] == M.shape[1]:
        nnodes = M.shape[0]
    else:
        raise ValueError("Matrix M must be square and symmetric")
    
    if M.dtype is not np.dtype('float64'):
        try:
            M=M.astype(np.float64)
        except:
            raise TypeError("M must be of numeric type, preferred float")    

    
    #Initial conditions
    #ic = np.ones((1, nnodes)) * np.array([0.131,  0.171, 0.343, 0.21, 3.07, 2.96,  25.36, 2.42])[:, None]
    #ic = (np.ones((1, nnodes, 8)) * np.random.normal(size=(1, nnodes, 8)) * np.array([10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0])).reshape(8, nnodes)
    
    # Generate random values between 0 and 1
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
    
    #f1.recompile()

    set_seed(seed); #Set the random seed

    if verbose == True:
        for i in range(1,Nsim):
            y_temp += dt * f1(y_temp, i, M, p, A,B, a,b,ad)
            #This line is for store values each dws points
            if (i % downsamp) == 0:
                y[i//downsamp,:,:] = y_temp
            if (i % (10 / dt)) == 0:
                print('Elapsed time: %i seconds'%(i * dt)) #this is for impatient people
    else:
        for i in range(1,Nsim):
            y_temp += dt * f1(y_temp, i, M, p,A,B, a,b,ad)
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



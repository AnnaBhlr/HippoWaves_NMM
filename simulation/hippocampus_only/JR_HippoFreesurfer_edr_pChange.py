#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:11:26 2024

@author: abehler
"""

import importlib
import JansenRitModel_8equations as JR8
import numpy as np
import os
# --- repo root (added during repo assembly: resolves data/ paths) ---
_REPO = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_REPO, 'data')) and os.path.dirname(_REPO) != _REPO:
    _REPO = os.path.dirname(_REPO)
from scipy.io import savemat
import coupling
import pygeodesic.geodesic as geodesic


importlib.reload(JR8)

def save_to_mat(timeVector, y, JR8):
    rounded_max = round(max(JR8.p), 2)
    p_str_max = str(rounded_max)
    p_str_max = p_str_max.replace('.', 'p')
    rounded_min = round(min(JR8.p), 2)
    p_str_min = str(rounded_min)
    p_str_min = p_str_min.replace('.', 'p')
    out_dir = os.path.join(_REPO, "output")  # results are written here
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"slowTheta_p_{p_str_min}_{p_str_max}_gaussnoise.mat")

    # Check if file exists and append a unique identifier if it does
    counter = 0
    original_filename = filename
    while os.path.exists(filename):
        counter += 1
        filename = f"{original_filename[:-4]}_{counter}.mat"


    output_dict = {
        'time_vector': timeVector,
        'pyrm': JR8.C2 * y[:,1] - JR8.C4 * y[:,2] + JR8.C * JR8.alpha * y[:,3],
        'metadata': {
            'p': JR8.p,
            'C1': JR8.C1,
            'C2': JR8.C2,
            'C3': JR8.C3,
            'C4': JR8.C4,
            'alpha': JR8.alpha,
            'beta': JR8.beta,
            'A': JR8.A,
            'B': JR8.B,
            'a': JR8.a,
            'b': JR8.b,
            'ad': JR8.ad,
            'sigma':JR8.sigma,
            'output':y
        }
    }
    savemat(filename, output_dict)
    

hippoMeshFile = os.path.join(_REPO, "data", "meshes", "hippocampus.vtk")
edr_lambda = 0.2  # EDR decay constant
KijFileName = os.path.join(_REPO, "data", "coupling", "Kij_norm_FS_exp0p2.mat")


if coupling.checkIfVtk(hippoMeshFile):
    import pyvista as pv
    mesh = pv.read(hippoMeshFile)
    verticesHippo = mesh.points
    facesHippo = mesh.faces
    facesHippo = facesHippo.reshape((-1, 4))[:, 1:]
    del mesh
    hippoKij = coupling.generateCouplingMatrix(verticesHippo, facesHippo, edr_lambda)
    hippoKij.tofile(KijFileName, sep =',')
 
    

geoalg = geodesic.PyGeodesicAlgorithmExact(verticesHippo, facesHippo)
D, _ = geoalg.geodesicDistances(np.array([376])) # idx of point at the tail/posterior part

D_max = np.max(D[D != np.inf])
print(f'max distance on mesh from tail {D_max}')
x = [0, D_max]


# set simulation parameters
JR8.dt = 1E-3 #Integration step
JR8.teq = 0 #Simulation time for stabilizing the system in s
JR8.tmax = 10#Length of simulated signals in s
JR8.downsamp = 10 #Downsampling to reduce the number of points   
C = 135
JR8.C1 = C * 1 #Connectivity between pyramidal pop. and excitatory pop.
JR8.C2 = C * 0.8 #Connectivity between excitatory pop. and pyramidal pop.
JR8.C3 = C * 0.25 #Connectivity between pyramidal pop. and inhibitory pop.
JR8.C4 = C * 0.25 #Connectivity between inhibitory pop. and pyramidal pop.

JR8.sigma = 0.1 #in 1/s (random uniform)
JR8.nnodes = hippoKij.shape[0] #number of nodes
JR8.seed = 0

# fast theta
#JR8.A = 5.6 
#JR8.B = 21
#JR8.a = 80 #in 1/s
#JR8.b = 70 #in 1/s
#JR8.ad = 90
#JR8.alpha = 0.55 #in 1/s
#JR8.beta = 0.35#in 1/s
# slow theta
JR8.A = 7 
JR8.B = 15
JR8.a = 100 #in 1/s
JR8.b = 50 #in 1/s
JR8.ad = 50
JR8.alpha = 0.4 #in 1/s
JR8.beta = 0.25#in 1/s

deltas = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] # in %
#pBaseline = 2 # fast theta
pUncus = 2.4 # slow theta

for delta in deltas:
    print(f'Processing delta {delta}%')
    
    pTail = pUncus + pUncus*(delta/100) # aka the posterior end
    y_p = [pTail, pUncus] 
    
    coefs = np.polyfit(x, y_p, 1)
    p = coefs[0] * D + coefs[1]
    JR8.p = p.squeeze()
    
    #init1 = time.time()
    y, timeVector = JR8.Sim(hippoKij,verbose = True)
    fieldact = JR8.C2 * y[:,1] - JR8.C4 * y[:,2] + JR8.C * JR8.alpha * y[:,3] #EEG-like output of the model #local field activity

    # save results and model parameter
    save_to_mat(timeVector, y, JR8)
    
    #end1 = time.time()
    #print([end1 - init1])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this simulation, coupling between hippocampus and cortex starts at t=0,
i.e., random conditions. Both systems are kept in the same frequency regime.
delta p and coulping between systems are subject to parameter sweeps.

@author: abehler
"""
import datetime
import importlib
import JansenRitModel_8equations_Mt as JR8
import math
import numpy as np
import pandas as pd
import pygeodesic.geodesic as geodesic
import pyvista as pv
import time
import os
# --- repo root (added during repo assembly: resolves data/ paths) ---
_REPO = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_REPO, 'data')) and os.path.dirname(_REPO) != _REPO:
    _REPO = os.path.dirname(_REPO)
import nibabel as nib
from scipy.io import savemat

importlib.reload(JR8)


def generateCouplingMatrix(vertices, faces, edr_lambda):
    """Exponential distance-rule (EDR) coupling matrix:
    Kij = exp(-edr_lambda * geodesic_distance(i, j)), row-normalised."""
    init1 = time.time()
    # Initialise the PyGeodesicAlgorithmExact class instance
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

    numberNodes = vertices.shape[0]
    Kij = np.zeros((numberNodes, numberNodes))

    # Fill the matrix
    for i in range(numberNodes):
        if i % 100 == 0:
            print(i)
        D, _ = geoalg.geodesicDistances(np.array([i]), None)

        for j in range(numberNodes):
            if D[j] > 0:
                Kij[i, j] = np.exp(-edr_lambda * D[j])
                Kij[j, i] = Kij[i, j]  # Fill the symmetric value

    Kij_norm = row_normalisation(Kij)
    end = time.time()
    print(f"Kij generated in {(end-init1)/60} min")
    return Kij_norm

def row_normalisation(Kij):
    Kij_norm = Kij / (np.sum(Kij, axis=1)[:, np.newaxis] * np.ones(Kij.shape))
    return Kij_norm

def remove_medial_wall(vertices, faces, medialWallMask):
    new_vertices = vertices[medialWallMask]
    #Create a mapping from old vertex indices to new indices
    index_mapping = np.where(medialWallMask)[0]  # Gets the new indices of the original vertices
    
    # Identify faces to remove
    # Convert mask to apply to faces
    faces_mask = np.all(medialWallMask[faces], axis=1)  # Checks if all vertices of a face are kept
    remaining_faces = faces[faces_mask]  # Keep only faces where all vertices are kept
    
    # Adjust indices in remaining faces to match filtered vertices
    # Create an inverse mapping for the vertices that are kept
    inverse_mapping = np.zeros_like(medialWallMask, dtype=int)
    inverse_mapping[index_mapping] = np.arange(len(index_mapping))
    
    # Update the faces to new indices
    new_faces = inverse_mapping[remaining_faces]
    
    return new_vertices, new_faces

def combineCouplingMatrices(Kij_mesh, Kij_sphere, mesh2sphere, coupling_value_mesh_to_sphere, coupling_value_sphere_to_mesh):
    """
    Combine two coupling matrices with sparse coupling between them.

    Parameters:
    Kij_mesh (numpy.ndarray): The coupling matrix for the mesh.
    Kij_sphere (numpy.ndarray): The coupling matrix for the sphere.
    mesh2sphere (list or numpy.ndarray): Mapping from mesh to sphere vertices.
    sphere2mesh (list or numpy.ndarray): Mapping from sphere to mesh vertices.
    coupling_value_mesh_to_sphere (float): Coupling value from mesh to sphere.
    coupling_value_sphere_to_mesh (float): Coupling value from sphere to mesh.

    Returns:
    numpy.ndarray: The combined coupling matrix.
    """
    n_vertices_mesh = Kij_mesh.shape[0]
    n_vertices_sphere = Kij_sphere.shape[0]

    # Create empty mapping matrices
    sphere2mesh_matrix  = np.zeros((n_vertices_mesh, n_vertices_sphere))

    # Generate mapping matrices
    for i, j in enumerate(mesh2sphere):
        if j is not None and not math.isnan(j):
            
            sphere2mesh_matrix[int(j), i] = 1   # mesh to sphere

    # Create empty combined matrix
    Kij_combined = np.zeros((n_vertices_mesh + n_vertices_sphere, n_vertices_mesh + n_vertices_sphere))

    # Fill diagonal blocks with individual coupling matrices
    Kij_combined[:n_vertices_mesh, :n_vertices_mesh] = Kij_mesh
    Kij_combined[n_vertices_mesh:, n_vertices_mesh:] = Kij_sphere

    # Add mapping matrices to combined matrix
    Kij_combined[:n_vertices_mesh, n_vertices_mesh:] = sphere2mesh_matrix * coupling_value_sphere_to_mesh # this is hippo to hemi
    Kij_combined[n_vertices_mesh:, :n_vertices_mesh] = sphere2mesh_matrix.T * coupling_value_mesh_to_sphere # this is hemi to hippo

    return Kij_combined,sphere2mesh_matrix

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_to_mat(base_path, y, timestamp):
    directory = os.path.join(base_path, timestamp)
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"simulation_{timestamp}.mat")
    output_dict = {'y': y}
    savemat(filename, output_dict)

def save_to_csv(base_path, timeVector, fieldact, timestamp):
    directory = os.path.join(base_path, timestamp)
    os.makedirs(directory, exist_ok=True)
    time_vector_filename = os.path.join(directory, f"time_vector_{timestamp}.csv")
    fieldact_filename = os.path.join(directory, f"fieldact_{timestamp}.csv")
    pd.DataFrame(timeVector).to_csv(time_vector_filename, index=False)
    pd.DataFrame(fieldact).to_csv(fieldact_filename, index=False)

def save_files(base_path, y, timeVector, fieldact):
    timestamp = get_timestamp()
    save_to_mat(base_path, y, timestamp)
    save_to_csv(base_path, timeVector, fieldact, timestamp)
    return timestamp

def save_parameters(outputFolder, parameters_df):
    filename = os.path.join(outputFolder, 'simulation_parameters.csv')
    parameters_df.to_csv(filename)

# === Define files and directories

outputFolder = os.path.join(_REPO, "output")  # results are written here
os.makedirs(outputFolder, exist_ok=True)
system1Input = os.path.join(_REPO, "data", "meshes", "hippocampus.vtk")
system2Input = os.path.join(_REPO, "data", "meshes", "tpl-fsaverage_den-10k_hemi-L_pial.surf.gii")
labelsFile = os.path.join(_REPO, "data", "meshes", "fsaverage5_medial_wall_lh_masked.txt")
couplingMatrix_file = os.path.join(_REPO, "data", "coupling", "hemi2hippo.csv")

# === Define configs for simulation

# EDR

l_Hippo = 0.3 
l_Cortex = 0.15 

# theta regime (same for both systems)

theta_choice = 'slow'  # options: 'slow' or 'fast'

# gradient in external input 

pDeltaHippo = 20  # e.g. 20 = 20% more input at tail

# coupling strength hippocampus -> cortex

hippo2hemi = 0.5  # unidirectional coupling strength
hemi2hippo = 0

# onset and offset of coupling in s

tCouplingOn = 0
tCouplingOff = 10

# === Handle hippocampus mesh

mesh = pv.read(system1Input)
verticesHippo = mesh.points
facesHippo = mesh.faces
facesHippo = facesHippo.reshape((-1, 4))[:, 1:]
del mesh
hippoKij = generateCouplingMatrix(verticesHippo, facesHippo, l_Hippo)
hippoNumVertices = hippoKij.shape[0]

# === Handle cortex mesh

# load txt with labeling (0 or 1) for medial wall vertices
medialWallMask = []
with open(labelsFile, 'r') as file:
    for line in file:
        medialWallMask.append(int(line.strip()))
medialWallMask = np.array(medialWallMask, dtype=bool)

gii_data = nib.load(system2Input)
faces = gii_data.darrays[1].data
vertices = gii_data.darrays[0].data

[verticesHemi, facesHemi] = remove_medial_wall(vertices, faces, medialWallMask)
hemiKij = generateCouplingMatrix(verticesHemi, facesHemi, l_Cortex)
del gii_data, faces, vertices, medialWallMask
hemiNumVertices = verticesHemi.shape[0]

print(f"Number of vertices on hippocampal mesh: {hippoNumVertices}")
print(f"Number of vertices on hemisphere mesh: {hemiNumVertices}")

#  === Set simulation parameters

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
JR8.seed = 0
JR8.nnodes = hippoNumVertices + hemiNumVertices #number of nodes

# theta configs
theta_configs_dict = {
    'fast': {
        'name': 'fast',
        'A': 5.6,
        'B': 21,
        'a': 80,
        'b': 70,
        'ad': 90,
        'alpha': 0.55,
        'beta': 0.35,
        'pUncus': 2.0
    },
    'slow': {
        'name': 'slow',
        'A': 7.0,
        'B': 15,
        'a': 100,
        'b': 50,
        'ad': 50,
        'alpha': 0.4,
        'beta': 0.25,
        'pUncus': 2.4
    }
}

# Apply parameter settings
theta_config = theta_configs_dict[theta_choice]

for param in theta_config:
    if param not in ['name', 'pUncus']:
        valueHippo = theta_config[param]
        setattr(JR8, param, np.concatenate([
            np.full(hippoNumVertices, valueHippo),
            np.full(hemiNumVertices, valueHippo)
        ]))
pBaseHippo = theta_config['pUncus']
pHemi = pBaseHippo

# Create spatial gradient of p over the hippocampus
geoalg = geodesic.PyGeodesicAlgorithmExact(verticesHippo, facesHippo)
D, _ = geoalg.geodesicDistances(np.array([376])) # 376 is idx of point at the tail/posterior part
D_max = np.max(D[D != np.inf])
x = [0, D_max]

y_p = [pBaseHippo + pBaseHippo * (pDeltaHippo / 100), pBaseHippo]
coefs = np.polyfit(x, y_p, 1)
pHippo = coefs[0] * D + coefs[1]
pHippoFlat = pHippo.squeeze()

# Uniform p for cortex
pHemiConst = np.full(hemiNumVertices, pHemi)
JR8.p = np.concatenate([pHippoFlat, pHemiConst]).squeeze()

# Build coupling matrices
hippo2hemiList = np.loadtxt(couplingMatrix_file, delimiter=",")
combined_matrix1, _ = combineCouplingMatrices(hippoKij, hemiKij, hippo2hemiList, 0, 0)
combined_matrix2, _ = combineCouplingMatrices(hippoKij, hemiKij, hippo2hemiList, hippo2hemi, hemi2hippo)
combined_matrix1 = row_normalisation(combined_matrix1)
combined_matrix2 = row_normalisation(combined_matrix2)

# Simulate
y, timeVector = JR8.Sim(combined_matrix1, combined_matrix2, tCouplingOn, tCouplingOff, verbose=True)
fieldact = JR8.C2 * y[:, 1] - JR8.C4 * y[:, 2] + JR8.C * JR8.alpha * y[:, 3]
timestamp = save_files(outputFolder, y, timeVector, fieldact)

# Save parameters
params = {
    'time': timestamp,
    'theta_mode_hippo': theta_config['name'],
    'theta_mode_hemi': theta_config['name'],
    'delta': pDeltaHippo,
    'pTailHippo': pBaseHippo + pBaseHippo * (pDeltaHippo / 100),
    'pUncusHippo': pBaseHippo,
    'pHemi': pHemi,
    'TimeCouplingOn': tCouplingOn,
    'TimeCouplingOff': tCouplingOff,
    'Hippo2Hemi': hippo2hemi,
    'Hemi2Hippo': hemi2hippo
}
params_df = pd.DataFrame([params])
save_parameters(outputFolder, params_df)
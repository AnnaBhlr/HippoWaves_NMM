#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:04:27 2024

Here we use data from Leonie Borne (published in NeuroImage 2023), 
the projection of gradient I eigenmap within the left hippocampus onto the 
cortical surface (see Fig 6). The volumetric data was projected onto meshes before.

@author: abehler
"""
import matplotlib.pyplot as plt
from nilearn import datasets, surface, plotting
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def sort_indices_by_values(vector):
    '''
   Used to rank vertices by ascending gradient value for hippocampal–cortical matching. Stable sorting preserves the original vertex order when values are equal, including ties introduced by filling missing values from nearby vertices. Missing projection values must be filled before this function is called.
   '''
    return np.argsort(vector, kind="stable").tolist()

# meshes
hippoMeshFile = '/path/to/meshes/hippocampus.vtk'
mesh = pv.read(hippoMeshFile)
verticesHippo = mesh.points
hippoNumVertices = verticesHippo.shape[0]
del mesh

# load txt with labeling (0 or 1) for medial wall vertices
file_path = '/path/to/meshes/meshes/fsaverage5_medial_wall_lh_masked.txt'
medialWallMask = []
with open(file_path, 'r') as file:
    for line in file:
        medialWallMask.append(int(line.strip()))
medialWallMask = np.array(medialWallMask, dtype=bool)

fsaverage = datasets.fetch_surf_fsaverage()
pial_mesh = fsaverage.pial_left
vertices, _ = surface.load_surf_mesh(pial_mesh)
verticesCortex = vertices[medialWallMask,:]
hemiNumVertices = verticesCortex.shape[0] # with medial wall included

print(f"Number of vertices on the hippocampal mesh: {hippoNumVertices}")
print(f"Number of vertices on the inflated hemisphere mesh: {hemiNumVertices}")

# hipocampal texture
textureHippo = np.loadtxt(
    "/path/to/Coupling_FunctionalProjections/EigenvectorHippocampus.csv",
                 delimiter=",")

# cortex texture, note this is including medial wall
textureCortex = np.loadtxt(
    "/path/to/Coupling_FunctionalProjections/projection2cortex_fs10k.csv",
                 delimiter=",")
# remove medial wall
textureCortex = textureCortex[medialWallMask]

# Fill missing projection values using nearest finite spherical neighbours.
sphereFile = '/path/to/meshes/tpl-fsaverage_den-10k_hemi-L_sphere.surf.gii'
sphereVertices, _ = surface.load_surf_mesh(str(sphereFile))
sphereVertices = np.asarray(sphereVertices, dtype=float)[medialWallMask]
sphereVertices /= np.linalg.norm(sphereVertices, axis=1, keepdims=True)

valid = np.isfinite(textureCortex)
_, nearest = cKDTree(sphereVertices[valid]).query(sphereVertices[~valid])
textureCortex[~valid] = textureCortex[valid][nearest]

# sort the texture data
sortedIdxHippo = sort_indices_by_values(textureHippo)
sortedIdxCortex = sort_indices_by_values(textureCortex)

# Initialize the mapping arrays with None for no connection
hippo2hemi = [np.nan] * hippoNumVertices
hemi2hippo = [np.nan] * hemiNumVertices

# Create one-to-one mapping from hippocampus to hemisphere
for i in range(hippoNumVertices):
    corresponding_hemi_vertex = int(i * hemiNumVertices / hippoNumVertices)
    hippo2hemi[sortedIdxHippo[i]] = sortedIdxCortex[corresponding_hemi_vertex]
    hemi2hippo[sortedIdxCortex[corresponding_hemi_vertex]] = sortedIdxHippo[i]
    
np.savetxt("hemi2hippo.csv", hemi2hippo, delimiter=",", fmt='%f')

# Create a color map
n_colors = hippoNumVertices
color_map = plt.cm.get_cmap('viridis', n_colors)

# Define black using an RGBA tuple
black = (0, 0, 0, 1)

# Initialize color lists with RGBA tuples for black
hippoColors = [black] * hippoNumVertices
hemiColors = [black] * hemiNumVertices

# Assign colors to connected vertices
for i, hemi_vertex in enumerate(hippo2hemi):
        color = color_map(i)  # This returns an RGBA tuple
        hippoColors[i] = color
        if hemi_vertex < len(hemiColors):
            hemiColors[hemi_vertex] = color

# Plotting
fig = plt.figure(figsize=(14, 7))

# 3D scatter plot for hippocampus vertices
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(verticesHippo[:, 0], verticesHippo[:, 1], verticesHippo[:, 2], c=hippoColors, label='Hippocampus Vertices')
ax1.set_title('Hippocampus Mesh')

# 3D scatter plot for hemisphere vertices
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(verticesCortex[:, 0], verticesCortex[:, 1], verticesCortex[:, 2], c=hemiColors, label='Hemisphere Vertices')
ax2.set_title('Hemisphere Mesh')
# Enable interactive rotation
plt.ion()
plt.show()   
    
    
    
    
    
    
    
    

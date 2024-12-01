#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we use data from 

Borne, L., Tian, Y., Lupton, M. K., Van Der Meer, J. N., Jeganathan, J., Paton, B., Koussis, N., Guo, C. C., 
Robinson, G. A., Fripp, J., Zalesky, A., & Breakspear, M. (2023). Functional re-organization of 
hippocampal-cortical gradients during naturalistic memory processes. NeuroImage, 271, 119996. 
https://doi.org/10.1016/j.neuroimage.2023.119996

to calculate a coupling of the hippocampus NMM network to the cortical hemisphere NMM network. We use  the 
projection of gradient I eigenmap within the left hippocampus onto the cortical surface (see Fig 6). 

The volumetric data was projected onto meshes before doing this.

@author: Anna Behler
"""
import matplotlib.pyplot as plt
from nilearn import datasets, surface, plotting
import numpy as np
import pyvista as pv


def sort_indices_by_values(vector):
    '''
    # Example usage:
    vector = [2, 3, 1, 5, 4]
    sorted_indices = sort_indices_by_values(vector)
    print(sorted_indices)
    Output: [2, 0, 1, 4, 3]
   '''
    return sorted(range(len(vector)), key=vector.__getitem__)


# load hippocampus mesh
hippoMeshFile = '/path/to/mesh/hippocampus.vtk'
mesh = pv.read(hippoMeshFile)
verticesHippo = mesh.points
hippoNumVertices = verticesHippo.shape[0]
del mesh

# get the cortical mesh
fsaverage = datasets.fetch_surf_fsaverage()
pial_mesh = fsaverage.pial_left
verticesCortex, _ = surface.load_surf_data(pial_mesh)

# load txt with labeling (0 or 1) for medial wall vertices
file_path = '/path/to/file/fsaverage5_medial_wall_lh_masked.txt'
medialWallMask = []
with open(file_path, 'r') as file:
    for line in file:
        medialWallMask.append(int(line.strip()))
medialWallMask = np.array(medialWallMask, dtype=bool)

# remove medial wall in cortical mesh
verticesCortex = verticesCortex[medialWallMask,:]
hemiNumVertices = verticesCortex.shape[0] 

print(f"Number of vertices on the hippocampal mesh: {hippoNumVertices}")
print(f"Number of vertices on the hemisphere mesh: {hemiNumVertices}")

# hipocampal texture
textureHippo = np.loadtxt(
    "/path/to/file/EigenvectorHippocampus.csv",
                 delimiter=",")

# cortex texture, note this is still with medial wall
textureCortex = np.loadtxt(
    "/path/to/file/projection2cortex_fs10k.csv",
                 delimiter=",")
# get rid of the medial wall
textureCortex = textureCortex[medialWallMask]

# Plotting
fig = plt.figure(figsize=(14, 7))

# 3D scatter plot for hippocampus vertices
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(verticesHippo[:, 0], verticesHippo[:, 1], verticesHippo[:, 2], c=textureHippo, label='Hippocampus Vertices')
ax1.set_title('Hippocampus Mesh')

# 3D scatter plot for hemisphere vertices
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(verticesCortex[:, 0], verticesCortex[:, 1], verticesCortex[:, 2], c=textureCortex, label='Hemisphere Vertices')
ax2.set_title('Hemisphere Mesh')
# Enable interactive rotation
plt.ion()
plt.show()   
    
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
    
#np.savetxt("hippo2hemi.csv", hippo2hemi, delimiter=",", fmt='%f')
#np.savetxt("hemi2hippo.csv", hemi2hippo, delimiter=",", fmt='%f')

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
    
    
    
    
    
    
    
    

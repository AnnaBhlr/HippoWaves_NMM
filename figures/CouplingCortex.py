import os
# --- repo root (added during repo assembly: resolves data/ paths) ---
_REPO = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_REPO, 'data')) and os.path.dirname(_REPO) != _REPO:
    _REPO = os.path.dirname(_REPO)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:40:07 2025

@author: ab799
"""

import math
from nilearn import datasets, surface
import numpy as np
import pygeodesic.geodesic as geodesic
import pyvista as pv

# do you wanna see the plot (or save the image)? ('yes' or 'no')
visible = 'yes'

# Load the mesh
fsaverage = datasets.fetch_surf_fsaverage()
mesh = fsaverage.pial_left
vertices, faces = surface.load_surf_data(mesh)

#load txt with labeling (0 or 1) for medial wall vertices
file_path = os.path.join(_REPO, "data", "meshes", "fsaverage5_medial_wall_lh_masked.txt")
medialWallMask = []
with open(file_path, 'r') as file:
    for line in file:
        medialWallMask.append(int(line.strip()))
medialWallMask = np.array(medialWallMask, dtype=bool)

# get rid of the medial wall
vertices = vertices[medialWallMask,:]
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
faces = inverse_mapping[remaining_faces]

# Convert faces to the format expected by PyVista
# Assuming each face is a triangle (3 points per face)
faces_pyvista = np.hstack([np.full((faces.shape[0], 1), 3), faces]).ravel()
# Create a PyVista mesh
mesh = pv.PolyData(vertices, faces_pyvista)

# Define the vertex of interest and exponential distance rule
vertex_of_interest = 7468
decay_factor = 0.15

# Initialise the PyGeodesicAlgorithmExact class instance
geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

# Compute distances from the vertex of interest to all other vertices
distances, _ = geoalg.geodesicDistances(np.array([vertex_of_interest]), None)

# Compute distances and colors
colors = np.exp(-decay_factor * distances)
colors[colors <= 1e-3] = 0  # Mask near-zero values
scalars = colors / colors.max()  # Normalize for better visualization

# Add scalars to the mesh
mesh["colors"] = scalars

if visible == 'yes':
    plotter = pv.Plotter()
else:
    plotter = pv.Plotter(off_screen=True)
    
plotter.add_mesh(mesh, scalars="colors", cmap="PuBuGn", show_edges=True)

# Set camera position (mimicking MATLAB perspective as much as possible)
mesh.rotate_x(math.sin(np.radians(15)), inplace=True)
mesh.rotate_x(math.cos(np.radians(15)), inplace=True)
mesh.rotate_z(-130, inplace=True)

# add orientation: red = X-axis, green = Y-axis, blue arrow = Z-axis
axes = pv.Axes()
axes_actor = axes.axes_actor
_ = plotter.add_orientation_widget(
    axes_actor,
    viewport=(0, 0, 0.3, 0.5),
)

if visible == 'yes':
    # Show the plot
    plotter.show()
else:
    # Save the plot as a high-DPI JPG
    dpi = 300  # Desired DPI
    default_dpi = 96  # Standard DPI for screens
    scale = dpi / default_dpi  # Calculate scale factor
    # Save as a high-resolution image
    plotter.screenshot("cortex_plot.jpg", scale=scale)

np.savetxt("EDR-cortex4matlabPlot.csv", scalars, delimiter=",")
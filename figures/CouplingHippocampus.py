import os
# --- repo root (added during repo assembly: resolves data/ paths) ---
_REPO = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_REPO, 'data')) and os.path.dirname(_REPO) != _REPO:
    _REPO = os.path.dirname(_REPO)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:32:32 2025

@author: ab799
"""

import math
import numpy as np
import pygeodesic.geodesic as geodesic
import pyvista as pv

# do you wanna see the plot (or save the image)? ('yes' or 'no')
visible = 'yes'

# Load the mesh
hippoMeshFile = os.path.join(_REPO, "data", "meshes", "hippocampus.vtk")
mesh = pv.read(hippoMeshFile)  # Replace with your actual file path
vertices = mesh.points
faces = mesh.faces.reshape((-1, 4))[:, 1:]

# Define the vertex of interest and exponential distance rule
vertex_of_interest = 3060
decay_factor = 0.3

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

# Apply camera settings
bounds = mesh.bounds
mesh_center = [
    (bounds[0] + bounds[1]) / 2,
    (bounds[2] + bounds[3]) / 2,
    (bounds[4] + bounds[5]) / 2,
]
camera_position = mesh_center + np.array([100, -100, 80])  # Offset for isometric view
focal_point = mesh_center
view_up = [0, 0, 1]  # Z-axis up

# Correctly format camera position for PyVista
plotter.camera_position = [
    tuple(camera_position),  # Camera position
    tuple(focal_point),      # Focal point
    tuple(view_up),          # View up vector
]

# Set azimuth and elevation
plotter.camera.azimuth = -120
plotter.camera.elevation = 15

# Set aspect ratio and background color
plotter.set_scale(1, 1, 1)  # Equal scaling for all axes
plotter.background_color = "white"

# Enable lighting
plotter.enable_lightkit()

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
    plotter.screenshot("hippocampus_plot.jpg", scale=scale)

np.savetxt("EDR-hippocampus4matlabPlot.csv", scalars, delimiter=",")
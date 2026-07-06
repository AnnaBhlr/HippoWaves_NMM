#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow-potential analysis for output of a single simulation.

Reads one timestamped simulation-output folder (as written by
simulation/coupled/run_coupled_system.py), computes the phase-gradient flow
field and its Helmholtz-Hodge potential for the cortex and the hippocampus,
runs an SVD on the potential, and writes the flow-potential CSVs (+ diagnostic
plots) back into that folder.
"""

import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (analysis/ -> repo)
import sys
import pyvista as pv
import numpy as np
import pandas as pd
from scipy.signal import hilbert
import matplotlib.pyplot as plt
# wave_detection_methods is external code (originally from the
# traveling-waves codebase). It lives in ../external/ (repo-level external
# folder) -- add it to the path so the import works regardless of cwd.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external"))
from wave_detection_methods import *

# The SINGLE simulation-output folder to analyse
# One timestamped run produced by simulation/coupled/run_coupled_system.py,
# e.g. output/20260204_162512. Set this to the run you want to process.
sim_dir = os.path.join(_REPO, "output", "REPLACE_WITH_RUN_TIMESTAMP")
timestamp = os.path.basename(os.path.normpath(sim_dir))
output_dir = sim_dir

# Handle hippocampus mesh
system1Input = os.path.join(_REPO, "data", "meshes", "hippocampus.vtk")
mesh = pv.read(system1Input)
verticesHippo = mesh.points.astype(np.float64)
facesHippo = mesh.faces.reshape((-1, 4))[:, 1:].astype(np.int64)
hippoNodes = verticesHippo.shape[0]

cortexinfl_path = os.path.join(_REPO, "data", "meshes", "fsaverage5_hemi_L_pial_noMedialWall.vtk")
cortexinfl = pv.read(cortexinfl_path)
v_lh = cortexinfl.points
f_lh = cortexinfl.faces.reshape((-1, 4))[:, 1:]

print(f"Processing simulation: {sim_dir}")

# Load time vector and field activity data for this run
time = pd.read_csv(os.path.join(sim_dir, f"time_vector_{timestamp}.csv"), header=None).values.flatten()
fieldact = pd.read_csv(os.path.join(sim_dir, f"fieldact_{timestamp}.csv"), header=None).values

# Set relaxation time and process field activity
relax_time = 0.05  # relaxation time, in seconds
dt = np.mean(np.diff(time))
discard_idx = np.argmax(time >= relax_time)
if discard_idx:
    time = time[discard_idx:] - time[discard_idx]
    fieldact_cortex = fieldact[discard_idx:, hippoNodes:]
    fieldact_hippo = fieldact[discard_idx:, :hippoNodes]

time_coupled = time[time > 2]
idx_coupling = len(time) - len(time_coupled)

# Hilbert transform of field activity
phase_lh = np.unwrap(np.angle(hilbert(fieldact_cortex - np.mean(fieldact_cortex, axis=0), axis=0)), axis=0).T
phase_hippo = np.unwrap(np.angle(hilbert(fieldact_hippo - np.mean(fieldact_hippo, axis=0), axis=0)), axis=0).T
# Calculate coherence (order parameter) across time
R = np.abs(np.mean(np.exp(1j * phase_lh), axis=0))

# Compute spatial gradients and perform Helmholtz-Hodge decomposition
boundary_k_ring = -1
boundary_mask_lh = k_ring_boundary(v_lh, f_lh, k=boundary_k_ring)

v_lh = v_lh.astype(np.float64)
f_lh = f_lh.astype(np.int64)

# Pre-compute gradient operator and barycenters
gradient_operator_lh = igl.grad(v_lh, f_lh)
bc_coords_lh = compute_barycentric_coords(v_lh, f_lh)
phase_grad_lh = compute_phase_gradient(phase_lh, f_lh, bc_coords_lh, gradient_operator_lh)
phase_grad_norm_lh = (phase_grad_lh / np.linalg.norm(phase_grad_lh, axis=0)).T
U_lh = compute_helmholtz_hodge_decomposition_amg(-phase_grad_norm_lh, v_lh, f_lh)

# Perform Singular Value Decomposition on U_lh
U, S, Vt = np.linalg.svd(U_lh[:, idx_coupling:])
variance_explained = (S ** 2) / np.sum(S ** 2)

# Plot singular vectors and save the figure
fig, axes = plt.subplots(3, 2, figsize=(15, 12), gridspec_kw={'width_ratios': [2, 1]})
for i in range(3):
    ax = axes[i, 0]
    ax.plot(time_coupled, Vt[i, :])
    ax.set_title(f'Right Singular Vector {i+1} (Temporal Pattern) for Times > 2 s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

for i in range(3):
    ax = fig.add_subplot(3, 2, 2 * i + 2, projection='3d')
    scatter = ax.scatter(v_lh[:, 0], v_lh[:, 1], v_lh[:, 2], c=U[:, i], cmap='viridis', s=10, alpha=0.8)
    ax.view_init(elev=0, azim=180)
    ax.set_title(f'Left Singular Vector {i+1} (Spatial Pattern)')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.grid(False)
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "singular_vectors_plot_cortex.png"), dpi=300)
plt.close(fig)

# Save U_lh as a CSV file
U_lh_df = pd.DataFrame(U_lh)
U_lh_df.to_csv(os.path.join(output_dir, "FlowPotential_cortex.csv"), index=False)

# now the hippocampus
boundary_mask_lh = k_ring_boundary(v_lh, facesHippo, k=boundary_k_ring)

# Pre-compute gradient operator and barycenters
gradient_operator_hippo = igl.grad(verticesHippo, facesHippo)
bc_coords_hippo = compute_barycentric_coords(verticesHippo, facesHippo)
phase_grad_hippo = compute_phase_gradient(phase_hippo, facesHippo, bc_coords_hippo, gradient_operator_hippo)
phase_grad_norm_hippo = (phase_grad_hippo / np.linalg.norm(phase_grad_hippo, axis=0)).T
U_hippo = compute_helmholtz_hodge_decomposition_amg(-phase_grad_norm_hippo, verticesHippo, facesHippo)

# Perform Singular Value Decomposition on U_hippo
U, S, Vt = np.linalg.svd(U_hippo[:, idx_coupling:])
variance_explained = (S ** 2) / np.sum(S ** 2)

# Plot singular vectors and save the figure
fig, axes = plt.subplots(3, 2, figsize=(15, 12), gridspec_kw={'width_ratios': [2, 1]})
for i in range(3):
    ax = axes[i, 0]
    ax.plot(time_coupled, Vt[i, :])
    ax.set_title(f'Right Singular Vector {i+1} (Temporal Pattern) for Times > 2 s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

for i in range(3):
    ax = fig.add_subplot(3, 2, 2 * i + 2, projection='3d')
    scatter = ax.scatter(verticesHippo[:, 0], verticesHippo[:, 1], verticesHippo[:, 2], c=U[:, i], cmap='viridis', s=10, alpha=0.8)
    ax.view_init(elev=0, azim=180)
    ax.set_title(f'Left Singular Vector {i+1} (Spatial Pattern)')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.grid(False)
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "singular_vectors_plot_hippo.png"), dpi=300)
plt.close(fig)

# Save U_hippo as a CSV file
U_hippo_df = pd.DataFrame(U_hippo)
U_hippo_df.to_csv(os.path.join(output_dir, "FlowPotential_hippo.csv"), index=False)

print(f"Finished processing {sim_dir}")

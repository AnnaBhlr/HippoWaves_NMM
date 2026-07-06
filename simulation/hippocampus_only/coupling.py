#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:56:31 2024

@author: abehler
"""
import numpy as np
import os
from scipy.io import savemat
import pandas as pd
import pygeodesic.geodesic as geodesic
import time

import matplotlib.pyplot as plt


def checkIfGii(filemane):

    # Split the file path into root and extension
    root, extension = os.path.splitext(filemane)

    # Check if the file extension is .mat
    if extension == '.gii':
        return True
    else:
        return False

def checkIfVtk(filemane):

    # Split the file path into root and extension
    root, extension = os.path.splitext(filemane)

    # Check if the file extension is .mat
    if extension == '.vtk':
        return True
    else:
        return False

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

    Kij_norm = Kij / (np.sum(Kij, axis=1)[:, np.newaxis] * np.ones((numberNodes, numberNodes)))
    end = time.time()
    print(f"Kij generated in {(end-init1)/60} min")
    return Kij_norm


def combineCouplingMatrices(Kij_mesh, Kij_sphere, mesh2sphere, sphere2mesh, coupling_value_mesh_to_sphere, coupling_value_sphere_to_mesh):
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

    # Create empty combined matrix
    Kij_combined = np.zeros((n_vertices_mesh + n_vertices_sphere, n_vertices_mesh + n_vertices_sphere))

    # Fill diagonal blocks with individual coupling matrices
    Kij_combined[:n_vertices_mesh, :n_vertices_mesh] = Kij_mesh
    Kij_combined[n_vertices_mesh:, n_vertices_mesh:] = Kij_sphere

    # Sparse coupling between mesh and sphere
    for i, j in enumerate(mesh2sphere):
        if j is not None:
            Kij_combined[i, n_vertices_mesh + j] = coupling_value_mesh_to_sphere  # mesh to sphere

    for i, j in enumerate(sphere2mesh):
        if j is not None:
            Kij_combined[n_vertices_mesh + i, j] = coupling_value_sphere_to_mesh  # sphere to mesh

    return Kij_combined

def save_to_mat(timeVector, y, JR8, Kij1, Kij2,filename):
    
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
            'Kij1': Kij1,
            'Kij2': Kij2,
            'sigma':JR8.sigma,
            'output':y
        }
    }
    savemat(filename, output_dict, format='5', do_compression=True)
    
def save_to_csv(timeVector, y, JR8, Kij1, Kij2, original_filename):
    # Ensure .csv extension
    original_filename = original_filename if original_filename.endswith('.csv') else original_filename + '.csv'
    
    # Base filename without .csv extension
    base_filename = original_filename[:-4]
    
    # Save timeVector
    timeVector_filename = f"{base_filename}_timeVector.csv"
    pd.DataFrame({'time_vector': timeVector}).to_csv(timeVector_filename, index=False)
    print('time saved')

    # Calculate and save pyrm data
    pyrm = JR8.C2 * y[:, 1] - JR8.C4 * y[:, 2] + JR8.C * JR8.alpha * y[:, 3]
    pyrm_filename = f"{base_filename}_pyrm.csv"
    pd.DataFrame(pyrm).to_csv(pyrm_filename, index=False)
    print('pyrm saved')
  
    # Prepare and save metadata
    metadata_filename = f"{base_filename}_metadata.csv"
    metadata_dict = {
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
        'sigma': JR8.sigma,
    }
    pd.DataFrame([metadata_dict]).to_csv(metadata_filename, index=False)
    print('metadata saved')
    return timeVector_filename, pyrm_filename, metadata_filename


# Example usage
# Assuming JR8 and other parameters are properly defined
# save_to_csv(timeVector, y, JR8, Kij1, Kij2, 'stimulation_data.csv')



def plotPowerSpectrum(activity, dt):
    """
    Plot the power spectrum of neural activity.

    Parameters:
    - activity: 2D numpy array with shape (time points, neurons)
    - dt: Time step between measurements in seconds
    """
    # Number of time points
    N = activity.shape[0]
    
    # Sampling rate
    Fs = 1.0 / dt  # Ensure 'dt' is defined as the time step between measurements
    
    # Subtract the mean to remove the DC component for each neuron
    activity_detrended = activity - np.mean(activity, axis=0)
    
    # Calculate the FFT for each neuron's activity along each column
    Y = np.fft.fft(activity_detrended, axis=0)
    
    # Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued number of points N.
    P2 = np.abs(Y / N)
    P1 = P2[0:N//2+1, :]  # Adjust for Python indexing
    P1[1:-1, :] = 2*P1[1:-1, :]  # Multiply by 2 (except for the DC and the Nyquist components, if N is even)
    
    # Calculate the frequency vector for the single-sided spectrum
    f = Fs * np.arange(0, N//2+1) / N
    
    # Average the spectrum over all neurons
    P_avg = np.mean(P1, axis=1)  # This averages across all columns (neurons)
    
    # Plotting the averaged frequency spectrum
    plt.plot(f, P_avg)
    plt.title('Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')

    # You don't need to show the plot here if this function is part of a larger plotting routine
    # plt.show()

# Example usage:
# plotPowerSpectrum(fieldact_mesh, dt)
# plt.show()

    
    
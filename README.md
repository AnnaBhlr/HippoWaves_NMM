# HippoCortexWaves

Code to reproduce the simulations and analyses of the cortex–hippocampus
traveling-waves manuscript.

## Repository structure

```
HippoCortexWaves/
├── simulation/
│   ├── coupled/              # coupled cortex <-> hippocampus model + runner
│   │   ├── JansenRitModel_8equations_Mt.py    # neural-mass model (time-switchable coupling)
│   │   └── run_coupled_system.py              # single simulation runner (builds EDR coupling inline)
│   └── hippocampus_only/     # hippocampus-only model + runner
│       ├── JansenRitModel_8equations.py       # neural-mass model (single static coupling)
│       ├── coupling.py                        # builds the hippocampal EDR coupling matrix
│       └── JR_HippoFreesurfer_edr_pChange.py  # single simulation runner
├── analysis/
│   ├── FlowPotential_CortexAndHippo.py   # flow potentials / phase gradients (one sim run)
│   ├── analyze_single_simulation.m       # one hippo + one coupled run -> R, speed, angle (single figure)
│   ├── compute_coherence.m               # Kuramoto order parameter R (extracted from figures)
│   ├── compute_velocity_field.m          # traveling-wave velocity field (extracted from figures)
│   └── compute_velocity_angle.m          # mean velocity angle to A-P (Y) axis (extracted from meanVelocityAngle.m)
├── figures/                  # manuscript figure scripts (MATLAB + Python)
├── external/                 # third-party vendored code (NOT ours) — see external/README.md
├── carlos_coronel_original/  # original Jansen-Rit model by C. Coronel (source of our models)
├── resampling_principal_gradient/  # builds the cortical principal gradient input
├── example_data/             # bundled minimal example run (analysis runs after clone)
├── data/                     # static inputs (meshes, coupling matrices, colormaps)
└── LICENSE                   # MIT
```

## Dependencies (must be installed separately)

### Python — simulations + the `FlowPotential` analysis
Python 3.x with:

| Package | Used for |
|---------|----------|
| `numpy`, `scipy`, `pandas`, `matplotlib` | core numerics / I/O / plotting |
| `numba` | JIT acceleration of the neural-mass integration |
| `pyvista` | surface-mesh reading (`.vtk`) |
| `nibabel` | reading GIFTI/CIFTI surfaces (`.gii`, `.nii`) |
| `nilearn` | surface / neuroimaging utilities |
| `pygeodesic` | geodesic distances on the mesh (coupling construction) |
| `libigl` (imported as `igl`), `potpourri3d`, `pyamg` | used by `external/wave_detection_methods.py` (Helmholtz–Hodge decomposition, geodesics) |

```
pip install numpy scipy pandas matplotlib numba pyvista nibabel nilearn pygeodesic libigl potpourri3d pyamg
```

### MATLAB — figure scripts in `figures/`
The figure scripts `addpath(...)` several **external MATLAB toolboxes that are not
included in this repo**. Install each and point the `addpath` lines at your local
copy (the paths in the scripts are currently machine-specific — see note below):

| Toolbox | Used for | Source |
|---------|----------|--------|
| **neural-flows** | optical-flow / flow-field analysis on the surface | https://github.com/brain-modelling-group/neural-flows |
| **brainwaves** | traveling-wave detection/analysis | obtain from its original distribution |
| **Scientific Colour Maps** (Crameri) | perceptually-uniform colormaps | https://www.fabiocrameri.ch/colourmaps/ |
| **slanCM** | colormap collection | MATLAB File Exchange (slandarer's slanCM) |
| **cnem_25-05-23** | mesh utilities `addpath`'d by some scripts | obtain from its original source |

## Pipeline

1. **Simulate** — run one of the two single-run scripts (no parameter sweeps):
   `simulation/coupled/run_coupled_system.py` (coupled cortex↔hippocampus) or
   `simulation/hippocampus_only/JR_HippoFreesurfer_edr_pChange.py` (hippocampus only).
   Each builds its EDR coupling on the fly and writes its results to a repo-local
   `output/` folder (created automatically; git-ignored) — e.g.
   `simulation_<timestamp>.mat`, `fieldact_<timestamp>.csv`, `time_vector_<timestamp>.csv`.
2. **Analyze** — run `analysis/FlowPotential_CortexAndHippo.py` on **one**
   simulation run (set `sim_dir` near the top to the run's `output/<timestamp>/`
   folder) to produce the flow-potential CSVs (`FlowPotential_cortex.csv`,
   `FlowPotential_hippo.csv`). It imports `external/wave_detection_methods.py`
   automatically (path is resolved relative to the script).
   Two further analysis quantities are provided as standalone MATLAB functions,
   extracted from the figure scripts so the calculations live with the analysis
   code: `analysis/compute_coherence.m` (Kuramoto order parameter R) and
   `analysis/compute_velocity_field.m` (traveling-wave velocity vector field; needs
   the neural-flows / brainwaves / cnem toolboxes — see Dependencies → MATLAB).

   `analysis/analyze_single_simulation.m` is a complete, self-contained MATLAB
   script that takes **one hippocampus-only run and one coupled run** (the `simDir`
   paths at the top default to the bundled example runs), reads their data, and
   produces a **single figure** comparing the coherence R, the mean phase speed, and
   the mean velocity angle over time. All three reproduce the manuscript
   calculations: R and speed as in `getVelocityHippocampus.m`, and the angle as the
   **mean angle to the +Y (anterior–posterior) axis** `<acos(v_y/|v|)>_nodes`
   (radians 0–π) as in `meanVelocityAngle.m`, via `compute_velocity_angle.m`.
   Also needs the neural-flows / brainwaves / cnem toolboxes.
3. **Figures** — run the scripts in `figures/` (mostly MATLAB) on the simulation
   and analysis output.

## Example data

A **minimal example is bundled in the repo** under `example_data/` — a ~1 s slice
(100 time points) of one hippocampus run and one coupled run, just the
`time_vector_*.csv` + `fieldact_*.csv` the analysis needs (~30 MB total, every
file < 25 MB).
`analysis/analyze_single_simulation.m` points at these by default, so it runs
straight after cloning (given the MATLAB toolboxes — see Dependencies).

The **full example runs** (~1.7 GB; the coupled `simulation_*.mat` alone is ~1.2 GB)
exceed GitHub's 100 MB per-file limit and are **not** in the repo (`out/` is
git-ignored). Host them externally (e.g. Zenodo / OSF / figshare) and add the
download link here; point `simHippoDir` / `simCoupledDir` at them to analyse
complete runs.

## Attribution

- The neural-mass models are **derived from Carlos Coronel's** Jansen-Rit
  implementation (`carlos_coronel_original/`). See each model file's header.
- `external/` contains code from the traveling-waves project (author: Dominik
  Koller) used by the flow-potential analysis. See `external/README.md`.
- `resampling_principal_gradient/hcp.embed.all.179.lh.dscalar.nii` is **not
  ours** — it is the HCP principal functional-connectivity gradient embedding
  (left hemisphere, 179 HCP subjects) from the Margulies et al. (2016, PNAS)
  `gradient_analysis` repository, at:
  `gradient_data/templates/hcp.embed.all.179.lh.dscalar.nii`
  <https://github.com/NeuroanatomyAndConnectivity/gradient_analysis/blob/master/gradient_data/templates/hcp.embed.all.179.lh.dscalar.nii>
  (The companion mesh `Q1-Q6_R440.L.midthickness.32k_fs_LR.surf.gii` in that
  folder is from the same repo's `gradient_data/templates/`.)

## ⚠️ Before publishing / running — required fixes

The **simulations** and the **`FlowPotential` analysis** are clone-and-run: all
input data are included under `data/` and `resampling_principal_gradient/`, and
both simulation runners build their EDR coupling matrices on the fly (no
precomputed coupling files needed). Only the figure scripts still need attention:

1. **Remaining machine-specific paths.** Input-data paths have already been
   repointed to this repo's `data/` (and `resampling_principal_gradient/`)
   folders. Two kinds of absolute paths were intentionally left and still need
   setting for your machine:
   - **Output directories.** The two simulation runners and the
     `FlowPotential` analysis now use the repo-local `output/` folder by default
     (sims write `output/<timestamp>/...`; the analysis reads those subfolders).
     The **MATLAB figure scripts** (`figures/*.m`) still read simulation/analysis
     output from, and save figures to, machine-specific locations (e.g. the iCloud
     `manuscript_HippoWaves/...` paths). Point those at `output/` (or wherever your
     results live) before running them.
   - **MATLAB toolbox `addpath` lines** — point them at your local installs of the
     toolboxes listed under **Dependencies → MATLAB** above.

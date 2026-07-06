# external/

Third-party code that is **not** ours but was used in our analysis. Vendored
here verbatim for reproducibility. All credit belongs to the original authors.

| File | Source | Used by | Original author |
|------|--------|---------|-----------------|
| `wave_detection_methods.py` | `travelingwaves_code_rev1/modules/wave_detection_methods.py` — from the paper *"Human connectome topology directs cortical traveling waves and shapes frequency gradients"* | `../analysis/FlowPotential_CortexAndHippo.py` (Helmholtz–Hodge decomposition, phase gradients, barycentric coords, k-ring boundary) | Dominik Koller |

The original Jansen-Rit model by Carlos Coronel — the source our simulation model
was derived from — lives in `../carlos_coronel_original/` (see that folder's README).

Please cite the original works when using anything in this folder.

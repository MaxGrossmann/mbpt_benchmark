# Documentation for Questaal database entries (JSON files in `questaal_database`)

These JSON files store results from density-functional theory calculations and many-body perturbation theory calculations performed using Questaal, formatted in a standardized way compatible with the [pymatgen](https://pymatgen.org/) library. Each file represents a single computed entry (i.e., material, calculation metadata, and calculation results).


---

## Top-level structure

We document only the fields used in our workflow. The [ComputedEntry](https://pymatgen.org/pymatgen.entries.html#pymatgen.entries.computed_entries.ComputedEntry) class provides a wide range of functionalities, but we only use a subset. Note that the workflow normally produces `ComputedStructureEntries` when run, but since we cannot provide the ICSD crystal structures, we converted them from `ComputedStructureEntries` to `ComputedEntries` (see the Disclaimer in `README.md`).

- **`entry_id`** *(str)*  
  A unique identifier for the entry, encoding the composition, the ICSD source database and the structure size.    
  Example: `"AgCl_icsd_64734_nsites_2"`

- **`composition`** *(dict)*  
  Elemental composition in fractional units. The corresponding crystal structure can be found in the ICSD using the ID provided in the entry_id field. 
  Example: `{ "Ag": 1.0, "Cl": 1.0 }`

- **`parameters`** *(dict)*  
  Dictionary containing calculation settings.

- **`data`** *(dict)*  
  Dictionary containing calculation results.

---

## Parameters

- **`dft_kppa_init`, `qsgw_kppa_init`**: Initial k-points per atom.  
- **`dft_tol`**: Total energy convergence threshold in Ry.  
- **`qsgw_tol`**: Band gap convergence threshold in Ry.  
- **`eps_tol`**: Dielectric function convergence threshold defined through a similarity coefficient (see [Phys. Rev. Mater. 8, L122201 (2024)](https://doi.org/10.1103/PhysRevMaterials.8.L122201)).  
- **`metal_flag_*`**: Boolean flags indicating whether a material is metallic in various calculation steps.  
- **`gmax`**: Energy cutoff parameter (see the Questaal documentation for details).  
- **`*_kpts`**: k-point grids used for Brillouin zone sampling in various calculation steps. Given as a list of integers, e.g., `[6,6,6]`.  
- **`*_conv_error_flag`**: Flags given as boolean integers indicating convergence issues (`0 = converged`, `1 = error`).  
- **`vbm_idx`**: Zero-based index of the valence band maximum.  
- **`nv`, `nc`**: Number of valence and conduction bands included in the BSE Hamiltonian.  
- **`*_max_iter`**: Maximum number of allowed self-consistency iterations for various calculations.
- **`finish`**: Boolean flag indicating whether the full workflow completed successfully. 

---

## Data

- **`*_conv_data`**: Convergence data stored as a list of lists for various workflow steps (see code for details).  
- **`*_time`**: Timing data in seconds.  
- **`gap_*`**: Band gaps (in eV) obtained from different computational methods.  
- **`bs_*`**: Band structure data from various computational methods, given as a dictionary:
  - `n_colors`, `n_bands`: Dimensions of the dataset.  
  - `tick_labels`: High-symmetry k-point labels (e.g., `Γ`, `X`, `L`, `W`).  
  - `bs_paths`: List of segments with `label`, `nk`, and `k_points` (array of fractional coordinates).  
  - `bands`: 2D array of eigenvalues (`nk`, `n_bands`) given in eV. 
- **`dos_*`**: Density of states data from various calculation methods (dictionary). Energies are given in eV, and arrays typically include total and projected DOS values.  
- **`eps_*`**: Dielectric function data calculated in the independent-particle approximation (dictionary). Contains real and imaginary parts of the dielectric tensor as functions of frequency (given in eV).

---

## Loading a JSON file using Python

```python
import json
from pymatgen.entries.computed_entries import ComputedEntry

with open("questaal_database/AgCl_icsd_64734_nsites_2.json") as f:
    data = json.load(f)

entry = ComputedEntry.from_dict(data)

print("Entry ID:", entry.entry_id)
print("Self-energy k-point grid:", entry.parameters["qsgw_kpts"])
print("LDA band gap (eV):", entry.data["gap_lda"])
print("QSGW band gap (eV):", entry.data["gap_qsgw"])
print("QSGW band structure dictionary keys:", entry.data["bs_qsgw"].keys())
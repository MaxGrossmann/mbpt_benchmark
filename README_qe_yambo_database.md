# Documentation for QE-Yambo database entries (JSON files in `qe_yambo_database`)

These JSON files store results from density-functional theory calculations and many-body perturbation theory performed using Quantum ESPRESSO (QE) and Yambo, formatted in a standardized way compatible with the [pymatgen](https://pymatgen.org/) library. Each file represents a single computed entry (i.e., material, calculation metadata, and calculation results).

---

## Top-level structure

We document only the fields relevant for our workflows. The [ComputedEntry](https://pymatgen.org/pymatgen.entries.html#pymatgen.entries.computed_entries.ComputedEntry) class provides a wide range of functionalities, but we only use a subset. Note that the workflow normally produces `ComputedStructureEntries` when run, but since we cannot provide the ICSD crystal structures, we converted them from `ComputedStructureEntries` to `ComputedEntries` (see the Disclaimer in `README.md`).

- **`entry_id`** *(str)*  
  A unique identifier for the material, determined by the ICSD ID of the crystal structure.     
  Example: `"484"`

- **`composition`** *(dict)*  
  Elemental composition in fractional units. The corresponding crystal structure can be found in the ICSD using the ID provided in the entry_id field.  
  Example: `{ "In": 1.0, "Bi": 2.0, "S": 4.0, "Cl": 1.0 }`

- **`parameters`** *(dict)*  
  Dictionary containing the convergence settings and results from the QE and Yambo workflows. The `data` entry was not used in this work (decision by Marc Thieme).

---

## Parameters

The `parameters` dictionary contains general information about the system, the convergence data, and the results. It also includes several sub-dictionaries named after individual workflow substeps (e.g., `qe_conv_lda`, `bg_conv_lda`, `gw_conv_lda`), which store parameters and results specific to subworkflow. This structure enables the tracing of all intermediate results of the combined QE-Yambo workflow within a single entry.

Further details about the stored quantities and their meaning can be found in the corresponding workflow scripts located in the `g0w0_benchmark/src/workflows/` directory.

### General information

- **`id`**: ICSD material identifier.  
- **`name`**: Chemical formula of the material.  
- **`ibrav`**: Bravais lattice index as used by pw.x in Quantum ESPRESSO. 
- **`vol`**: Cell volume in Å³.  
- **`num_elec_*`**: Number of electrons (this depends on the pseudopotential and can differ between LDA and PBE calculations).   
- **`max_mem`**: Maximum recorded memory usage in MB.  
- **`lda_pw_cutoff_Ry`**: Initial plane-wave cutoff energy used for the material in Ry.  

### Available subworkflows

The QE-Yambo workflow consists of the following substeps, split into the LDA and PBE branches. Each substep corresponds to a specific script located in the `g0w0_benchmark/src/workflows/` directory.

- **`qe_convergence_lda`**
- **`bandgap_convergence_lda`**
- **`yambo_g0w0_conv_lda`**
- **`yambo_g0w0_ppa_lda`**

- **`qe_convergence_pbe`**
- **`bandgap_convergence_pbe`**
- **`yambo_g0w0_conv_pbe`**
- **`yambo_g0w0_ppa_pbe`**

---

## Loading a JSON file using Python

```python
import json
from pymatgen.entries.computed_entries import ComputedEntry

with open("qe_yambo_database/484.json") as f:
    data = json.load(f)

entry = ComputedEntry.from_dict(data)

print("Material:", entry.parameters["name"])
print("LDA band gap (eV):", entry.parameters["indirect_gap_lda"])
print("Direct G0W0@LDA gap (eV):", entry.parameters["g0w0_ppa_lda"]["gw_direct_gap"])
print("Number of bands used in the G0W0@LDA:", entry.parameters["gw_conv_lda"]["gw_bands"])
print("Complex Z-factors from an G0W0@LDA calculation (band index, Re, Im):", entry.parameters["g0w0_ppa_lda"]["z_factor"])

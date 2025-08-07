# Automated QSGW/QSGW^ calculations for non-magnetic semiconductors and insulators

This repository contains a workflow that automatically performs LDA → QPG<sub>0</sub>W<sub>0</sub> → QSGW → QSGW^ calculations for non-magnetic semiconductors and insulators using the Questaal code.
We ran this workflow for the materials in the Borlido et al. benchmark dataset[^jctc][^npj].

[^jctc]: P. Borlido, T. Aull, A. W. Huran, F. Tran, M. A. L. Marques, and S. Botti, Large-scale benchmark of exchange–correlation functionals for the determination of electronic band gaps of solids, J. Chem. Theory Comput. 15, 5069–5079 (2019), https://doi.org/10.1021/acs.jctc.9b00322
[^npj]: P. Borlido, J. Schmidt, A. W. Huran, F. Tran, M. A. L. Marques, and S. Botti, Exchange-correlation functionals for band gaps of solids: benchmark, reparametrization and machine learning, npj Comput. Mater. 6, 96 (2020), https://doi.org/10.1038/s41524-020-00360-0

We currently support both local calculations (`main_local.py`) and calculations on a Slurm-based supercomputer (`main_noctua.py` and `restart_noctua.py`), though the latter requires minor changes to `./qsgw_workflow/utils/sbatch.py`. 

The code was tested on the Paderborn Center for Parallel Computing (PC<sup>2</sup>) and supports single and multi-node parallelization.

**Installation:**

Questaal:

We used Questaal commit 397e0cb from the `dev` branch for the benchmark calculations.
Note that the workflow assumes you have added all Questaal binaries to your `PATH` variable.

Workflow:

```
# 1. clone
git clone https://github.com/MaxGrossmann/mbpt_benchmark.git
cd mbpt_benchmark/qsgw_benchmark

# 2. (optional) create an isolated python environment, i.e., using conda
conda create -n mbpt_benchmark python=3.12
conda activate mbpt_benchmark

# 3. install dependencies and workflows
pip install -e .
```

**Usage**:

To test the workflow, simply run `main_local.py`. 
The currently available structure can be found in the `structures/` directory.
The workflow performs calculations in the following order:

* LMTO basis setup
* LDA DFT k-point grid convergence
* LDA DFT self-consistency loop
* LDA DFT band structure
* LDA DFT DOS
* LDA DFT dielectric function in the independent particle approximation (IPA)
* QPG<sub>0</sub>W<sub>0</sub> k-point grid convergence 
* QPG<sub>0</sub>W<sub>0</sub> band structure
* QPG<sub>0</sub>W<sub>0</sub> DOS
* QPG<sub>0</sub>W<sub>0</sub> dielectric function in the independent particle approximation (IPA)
* QPG<sub>0</sub>W<sub>0</sub>+SOC band gap
* QSGW self-consistency loop
* QSGW band structure
* QSGW DOS
* QSGW dielectric function in the independent particle approximation (IPA)
* QSGW^ self-consistency loop
* QSGW^ band structure
* QSGW^ DOS
* QSGW^ dielectric function in the independent particle approximation (IPA)
* QSGW^+SOC band gap

The workflow is designed so that it can easily be restarted from any point as long
as the convergence criteria remain unchanged. If the convergence criteria are changed,
however, the workflow starts from scratch.

Through some adjustments to the files `./qsgw_workflow/utils/sbatch.py`, `main_noctua.py`, and `restart_noctua.py`, one can use the workflow on a Slurm-based cluster to easily calculate multiple materials in parallel. Parallelization over multiple nodes is supported. You can adjust the total number of cores you want to use simultaneously for all materials combined by changing the number inside `control/ncores`. The number of cores that each job uses can be adjusted in the file `main_noctua.py`.


**Input structures:**

The code uses ComputedStructureEntries (CSEs) from pymatgen as input. 
All CSE files are stored instored in the  `structures/` directory.
We decided to use CSEs because the majority of large material databases, such as the Alexandria database, use them. 
Additionally, the format is ideal for storing computational data. 
Unfortunately, we cannot publish all of the input structures because they are from the ICSD. 
Nevertheless, in accordance with the ICSD license agreement, we have provided five sample input files.
If you want to recalculate all the materials in the benchmark, you will need to set up the input JSON files yourself.
To do so, download all the materials in the benchmark dataset from the ICSD, then check out the notebook `cif2input.ipynb` in the top directory of the repository.
The ICSD IDs for all materials can be found in the publication by Borlido et al. [https://doi.org/10.1021/acs.jctc.9b00322].

**Plotting band structures, density of states, and dielectric functions:**

In the analysis directory, we included two notebooks, `check_db_entry.ipynb` and `check_bse_transition_space.ipynb`, that enable easy plotting of the LDA, QSGW, and QSGW^ band structure, density of states (DOS), and dielectric function. In particular, the notebook `check_bse_transition_space.ipynb` highlights the automated selection of the transition space for the BSE.

**Project layout:**

```
qsgw_benchmark
├── analysis
│   ├── audit_benchmark.ipynb
│   ├── check_bse_transition_space.ipynb
│   └── check_db_entry.ipynb
├── control
│   └── ncores
├── job_scripts
│   ├── run_noctua_finer_kgrid.py
│   └── run_noctua.py
├── qsgw_workflow
│   ├── files
│   │   ├── borlido.csv
│   │   └── plotstyle.mplstyle
│   ├── utils
│   │   ├── __init__.py
│   │   ├── helper.py
│   │   ├── runner.py
│   │   ├── sbatch.py
│   │   └── system_config.py
│   ├── workflows
│   │    ├── __init__.py
│   │    ├── base.py
│   │    └── semiconductor.py
│   └── __init__.py
├── structures
│   ├── benchmark
│   └── test
├── .gitignore
├── LICENSE.txt
├── main_local.py
├── main_noctua.py
├── pyproject.toml
├── README.md
└── restart_noctua.py
```

**Acknowledgments**

We would like to thank the Questaal developers, especially Jerome Jackson and Brian Cunningham, for their support and feedback.

**Licence**

Copyright (c) 2025 Max Großmann and Malte Grunert

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

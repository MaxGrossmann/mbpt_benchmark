# Automated G<sub>0</sub>W<sub>0</sub>-PPA calculations for non-magnetic semiconductors and insulators

This repository contains a workflow that automatically performs LDA/PBE → G<sub>0</sub>W<sub>0</sub>-PPA calculations for non-magnetic semiconductors and insulators using Quantum ESPRESSO for DFT calculations and Yambo for GW calculations.
This workflow provides a quick and precise method for calculating and converging G<sub>0</sub>W<sub>0</sub>-PPA band gaps of solids.
We ran this workflow for the materials in the Borlido et al. benchmark dataset[^jctc][^npj].

[^jctc]: P. Borlido, T. Aull, A. W. Huran, F. Tran, M. A. L. Marques, and S. Botti, Large-scale benchmark of exchange–correlation functionals for the determination of electronic band gaps of solids, J. Chem. Theory Comput. 15, 5069–5079 (2019), https://doi.org/10.1021/acs.jctc.9b00322
[^npj]: P. Borlido, J. Schmidt, A. W. Huran, F. Tran, M. A. L. Marques, and S. Botti, Exchange-correlation functionals for band gaps of solids: benchmark, reparametrization and machine learning, npj Comput. Mater. 6, 96 (2020), https://doi.org/10.1038/s41524-020-00360-0

**Installation:**

Ab initio codes:

We used Quantum ESPRESSO 7.1 and Yambo 5.2.4 for the benchmark calculations.
Note that the workflow assumes you have added all Quantum ESPRESSO and Yambo binaries to your `PATH` variable.

Workflow:

```
# 1. clone
git clone https://github.com/MaxGrossmann/mbpt_benchmark.git
cd mbpt_benchmark/g0w0_benchmark

# 2. (optional) create an isolated python environment, i.e., using conda
conda create -n mbpt_benchmark python=3.12
conda activate mbpt_benchmark

# 3. install dependencies and workflows
pip install -r requirements.txt
```

**Usage**:

To test the workflow, simply run `main_local.py`. The currently available test structures can be found in the directory `input/test/`. 
The workflow performs calculations in the following order:

* LDA DFT k-point grid and plane-wave cutoff energy convergence
* PBE DFT k-point grid and plane-wave cutoff energy convergence
* LDA DFT band gap convergence
* PBE DFT band gap convergence
* LDA G<sub>0</sub>W<sub>0</sub> convergence (see https://doi.org/10.1038/s41524-024-01311-9)
* PBE G<sub>0</sub>W<sub>0</sub> convergence (see https://doi.org/10.1038/s41524-024-01311-9)
* LDA G<sub>0</sub>W<sub>0</sub> correction calculation
* PBE G<sub>0</sub>W<sub>0</sub> correction calculation

By making some adjustments to the functions in `./src/basic_utils/start_calc_lsf.py` and `main_lsf.py`, you can use the workflow on a LSF-based cluster to easily calculate multiple materials in parallel. 
Parallelization over multiple nodes is not supported. 
You can adjust the total number of cores you want to use simultaneously for all materials combined by changing the number inside `control/ncores`.
The number of cores that each job uses can be adjusted in the file `main_lsf.py`.

**Input structures:**

The code uses ComputedStructureEntries (CSEs) from pymatgen as input. 
All CSEs for the benchmark are stored in a pickle file in the `input/` directory.
We decided to use CSEs because the majority of large material databases, such as the Alexandria database, use them. 
Additionally, the format is ideal for storing computational data. 
Unfortunately, we cannot publish all of the input structures because they are from the ICSD. 
Nevertheless, in accordance with the ICSD license agreement, we have provided a sample input file containing five materials.
If you want to recalculate all the materials in the benchmark, you will need to set up the input file yourself.
To do so, download all the materials in the benchmark dataset from the ICSD, then check out the notebook `cif2input.ipynb` in the top directory of the repository.
The ICSD IDs for all materials can be found in the publication by Borlido et al. [https://doi.org/10.1021/acs.jctc.9b00322].

**Project layout:**

```
g0w0_benchmark
├── control
│   └── ncores
├── input
│   ├── test
│   └── benchmark_structures.pkl
├── pseudo
│   ├── LDA
│   └── PBE
├── src
│   ├── utils
│   │   ├── init__.py
│   │   ├── basic_utils.py
│   │   ├── calc_data_class.py
│   │   ├── qe_helper.py
│   │   ├── qe_runner.py
│   │   ├── qe_write.py
│   │   ├── unit_conversion.py
│   │   ├── yambo_gw_conv_class.py
│   │   ├── yambo_helper.py
│   │   ├── yambo_runner.py
│   │   └── yambo_write.py
│   ├── workflows
│   │   ├── init__.py
│   │   ├── bandgap_convergence_lda.py
│   │   ├── bandgap_convergence_pbe.py
│   │   ├── qe_convergence_lda.py
│   │   ├── qe_convergence_pbe.py
│   │   ├── yambo_g0w0_conv_lda.py
│   │   ├── yambo_g0w0_conv_pbe.py
│   │   ├── yambo_g0w0_ppa_lda.py
│   │   └── yambo_g0w0_ppa_pbe.py
│   ├── init__.py
│   └── do_all_workflow.py
├── .gitignore
├── LICENSE.txt
├── main_local.py
├── main_lsf.py
├── README.md
└── requirements.txt
```

**Licence**

Copyright (c) 2025 Marc Thieme and Max Großmann

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

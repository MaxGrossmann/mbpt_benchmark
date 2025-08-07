# Many‑Body Perturbation Theory Benchmark

This repository contains two separate workflows that, when combined, perform various flavors of GW calculations. We used these workflows to calculate materials from the Borlido et al. benchmark dataset[^jctc][^npj]. The calculation results are provided as JSON files. A detailed analysis of the results, including all of the tables and figures shown in the associated publication, can be found in the `results.ipynb` notebook. A summary of all the results is available as an Excel spreadsheet in the `spreadsheets/` directory. An updated version of the Borlido benchmark dataset was also placed there and is named `revised_bandgap_benchmark.xlsx`. In the revised dataset, we removed materials with questionable experiments and updated some experimental values with more recent ones. We urge adoption of our revised dataset. Please refer to our publication for details regarding the removed or updated materials.

[^jctc]: P. Borlido, T. Aull, A. W. Huran, F. Tran, M. A. L. Marques, and S. Botti, Large-scale benchmark of exchange–correlation functionals for the determination of electronic band gaps of solids, J. Chem. Theory Comput. 15, 5069–5079 (2019), https://doi.org/10.1021/acs.jctc.9b00322
[^npj]: P. Borlido, J. Schmidt, A. W. Huran, F. Tran, M. A. L. Marques, and S. Botti, Exchange-correlation functionals for band gaps of solids: benchmark, reparametrization and machine learning, npj Comput. Mater. 6, 96 (2020), https://doi.org/10.1038/s41524-020-00360-0

**Repository structure:**

| Path                   | Purpose                                                                                                                     
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`cifs/`** | This directory contains five example CIF files from the ICSD of materials used in the benchmark. |
| **`csvs/`** | Band gap benchmark overview as simple CSV files. For future benchmarks, please use the **`revised_bandgap_benchmark.csv`** file. Refer to the publication for details. |
| **`figures/`**  | Contains all figures shown in the publication. |
| **`g0w0_benchmark/`** | An automated, G<sub>0</sub>W<sub>0</sub>-PPA workflow using Quantum ESPRESSO and Yambo. For a detailed description, check the `README.md` file in the directory. |
| **`qe_yambo_database/`** | Database created using the **`g0w0_benchmark/`** workflow. We removed structural information from the database entries to comply with the ICSD license. |
| **`qsgw_benchmark/`** | An automated workflow for different types of quasiparticle self-consistent GW calculations using the Questaal code. For a detailed description, check the `README.md` file in the directory. |
| **`questaal_database/`** | Database created using the **`qsgw_benchmark/`** workflow. We removed structural information from the database entries to comply with the ICSD license. |
| **`spreadsheets/`** | Band gap benchmark overview as simple Excel spreadsheets. For future benchmarks, please use the **`revised_bandgap_benchmark.xlsx`** file. Refer to the publication for details. |
| **`cif2input.ipynb`** | A Jupyter notebook that converts CIF files to the input formats needed by both workflows. |
| **`results.ipynb`** | A Jupyter notebook that loads calculation results from provided databases and generates every plot and table shown in the manuscript and Supplementary Information. |
---

**Installation:**

```
# 1. clone
git clone https://github.com/MaxGrossmann/mbpt_benchmark.git
cd mbpt_benchmark

# 2. (optional) create an isolated python environment, i.e., using conda
conda create -n mbpt_benchmark python=3.12
conda activate mbpt_benchmark

# 3. install dependencies and workflows
pip install -r requirements.txt
```

**Disclaimer:**

Unfortunately, we cannot publish all of the input structures because they are licensed by the ICSD. 
Therefore we just provide input for five materials from the benchmark in the workflow directories `g0w0_benchmark/input/` and `qsgw_benchmark/structures/benchmark/`.
If you want to recalculate all the materials in the benchmark, you will need to set up the workflow input files yourself.
To do so, download all the materials in the benchmark dataset from the ICSD. 
The ICSD IDs for all the materials, along with a guide on how to download the ICSD CIFs, can be found in the file `icsd_query.txt`.
Then, check out the Jupyter notebook `cif2input.ipynb`, which generates the input files for both workflows from CIF files.

We also removed all structural information from the database entries.
In other words, we converted them from `ComputedStructureEntries` (CSE) to `ComputedEntries` (CE) because we are not permitted to provide all ICSD structures.
Note that both workflows produce CSEs when run.

**Licence**

Copyright (c) 2025 Max Großmann, Marc Thieme, and Malte Grunert

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

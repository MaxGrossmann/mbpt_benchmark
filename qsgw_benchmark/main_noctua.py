"""
DESCRIPTION:
    This version is designed for high-throughput calculation on Noctua 2 (PC2 in Paderborn, Germany). This
    script will start calculations for all materials in the selected 'structures' subdirectory. Once every 
    five minutes the script checks if a new calculation can be started and if so, it starts it. We start with
    all materials with one site in the unit cell and then move to cells with two sites, and so on. The 'nnodes'
    and 'ncores_per_node' variables determine how many cores each job can use. The total number of available 
    coresthat the script can use can be adjusted on the fly in the 'ncores' file in the 'control/' directory. 
    If you want to stop the script, just use 'CTRL+C'.
    
    Currently, the script is set up to run materials for our MBPT benchmark. Unfortunately, we cannot publish
    all of the input structures because they are from the ICSD. Nevertheless, in accordance with the ICSD license 
    agreement, we have provided a sample input file containing five materials. If you want to recalculate all the 
    materials in the benchmark, you will need to set up the input file yourself. To do so, download all the materials
    in the benchmark dataset from the ICSD, then check out the 'cif2input.ipynb' notebook in the top directory of the
    repository.
    
    To test the workflow, please use the 'main_local.py' script.
"""

# external imports
import os
import sys
import time
import numpy as np

# internal imports
import qsgw_workflow.utils.sbatch as sbatch
from qsgw_workflow.utils.helper import load_db_entry

# make sure we start in the right directory and get the base path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
base_path = os.getcwd()

"""
START OF USER INPUT.
"""

# skip flag:
# if 'True', materials for which a calculation directory exists will be skipped
# if 'False', all calculations will be restarted
# (what exactly happens depends on the state of your database entry, see the 'BaseWorkflow' class)
# if you want to restart only certain calculations, please use the 'restart_noctua' script
skip = True

# number of nodes for each job
nnodes = 1

# number of cores per node for each job
ncores_per_node = 16

# available memory (GB) per node
memory_per_node = 128

# maximum wall time (h)
wall_time = 14 * 24 # two weeks

# maximum number of sites in unit cell
max_sites = 6

# queue (normal, largemem)
queue = "normal"

# path to the script we want to run for all materials (absolute path)
script_path = f"{base_path:s}/job_scripts/run_noctua.py"

"""
END OF USER INPUT.
"""

# path to a directory with CSE files (absolute path)
struct_path = f"{base_path:s}/structures/benchmark"

# path to a directory where all calculations will take place,
# e.g., a scratch directory (absolute path)
calc_path = f"{base_path:s}/../questaal_calc"

# path to a directory where the database file will be saved (absolute path)
db_path = f"{base_path:s}/../questaal_database"

# slurm job name
job_name = "benchmark"

# name of the conda environment to use
conda_env = "mbpt_benchmark"

# database directory setup
if not os.path.exists(db_path):
    os.makedirs(db_path)

# check that the structure directory exists
if not os.path.exists(struct_path):
    raise FileNotFoundError(f"The path to the 'structure' directory, '{struct_path:s}', does not exist. Please run 'cif2input.ipynb' first!")

# get a list of all CSE files and keep those with less than or equal to 'max_sites' sites
cse_files = [fname for fname in os.listdir(struct_path) if not os.path.isdir(fname) and fname.endswith(".json")]
if not cse_files:
    raise FileNotFoundError(f"No JSON files with CSEs were found in the directory {struct_path:s}!")
cse_files = np.array(cse_files)
num_sites = []
for cse_file in cse_files:
    cse = load_db_entry(os.path.join(struct_path, cse_file))
    num_sites.append(cse.structure.num_sites)
num_sites = np.array(num_sites)
sort_idx = np.argsort(num_sites) # we want to start the materials with the smallest number of sites first
num_sites = num_sites[sort_idx]
cse_files = cse_files[sort_idx]
cse_files = [fname.split(".")[0] for i, fname in enumerate(cse_files) if num_sites[i] <= max_sites] # split off the file ending...

# COMMENT THIS OUT FOR PRODUCTION RUNS
# materials suitable for workflow testing
# cse_files = [
#     "TlCl_icsd_5253_nsites_2",
#     "InP_icsd_41443_nsites_2",
#     "Ne_icsd_43427_nsites_1",
#     "Si_icsd_51688_nsites_2",
#     "CsAu_icsd_58427_nsites_2",
#     "TlI_icsd_60491_nsites_2",
#     "NaI_icsd_61502_nsites_2",
#     "LiF_icsd_62361_nsites_2",
#     "GaP_icsd_77087_nsites_2",
#     "CuCl_icsd_78270_nsites_2",
#     "NaCl_icsd_181148_nsites_2",
#     "Ge_icsd_184252_nsites_2",
#     "CdSe_icsd_187310_nsites_2",
#     "C_icsd_190650_nsites_2",
#     "BeSe_icsd_616419_nsites_2",
#     "BeTe_icsd_616439_nsites_2",
#     "ScN_icsd_644666_nsites_2",
# ]

# sanity check
ncores_per_job = nnodes * ncores_per_node
print(
    f"Do you really want to start calculations for {len(cse_files):d} materials using the following settings?"
)
print(f"    skip            = {str(skip):s}")
print(f"    nnodes          = {nnodes:d}")
print(f"    ncores_per_node = {ncores_per_node:d}")
print(f"    memory_per_node = {memory_per_node:d}")
print(f"    ncores_per_job  = {ncores_per_job:d}")
print(f"    wall_time       = {wall_time:d}")
print(f"    max_sites       = {max_sites:d}")
print(f"    queue           = {queue:s}")
print(f"    script_path     = {script_path:s}")
print("Y/N?")
user_input = input()
if user_input != "Y":
    sys.exit()
    
# list of materials that need to be run with a finer initial k-point grid
finer_kgrid = [
    "CsAu_icsd_58427_nsites_2", 
    "BaO_icsd_616005_nsites_2",
    "TiO2_icsd_92363_nsites_6",
]

# loop over all structures
idx = 0
while idx < len(cse_files):

    # current materials
    struct_name = cse_files[idx]

    # total number of cores we want/can use
    ncores_total = np.loadtxt("control/ncores", dtype=int)

    # find out how if any jobs can be started
    # (this is only works for the SBATCH job system of the PC2 supercomputer in Paderborn, Germany...)
    ncores_curr = sbatch.current_running_tasks(job_name)

    # check and print the current status
    max_jobs = ncores_total // ncores_per_job
    curr_jobs = ncores_curr // ncores_per_job
    print(f"\nCurrently {curr_jobs:d} out of {max_jobs:d} possible jobs are running.", flush=True)

    # create and submit a job (if possible)
    if curr_jobs < max_jobs:

        # skip starting a job if the job directory already exists and the skip flag is True
        current_calc_path = f"{calc_path:s}/{struct_name:s}"
        if os.path.exists(current_calc_path):
            if skip:
                print(f"\nSkipping {struct_name:s}...", flush=True)
                idx += 1
                continue
        os.makedirs(current_calc_path, exist_ok=True)

        # check if a database entry exists, if so, check if the workflow has already been completed
        db_file_path = os.path.join(db_path, struct_name + ".json")
        if os.path.exists(db_file_path):
            print(f"Found database entry for {struct_name:s}.")
            try:
                db_entry = load_db_entry(db_file_path)
                print("The parsing of the database entry was successful.")
                if db_entry.parameters["finish"] == True:
                    print(f"All calculations for {struct_name:s} have finished successfully, so it is skipped.")
                    idx += 1
                    continue
                else:
                    print(f"Restarting {struct_name:s}, not all calculations are finished.")
            except:
                print( f"The parsing of the database entry was unsuccessful, restarting {struct_name:s}.")

        # print the current calculation number and material
        with open("control/current_idx", "w") as f:
            f.write(
                f"Running material {struct_name:s} ({idx + 1:d} of {len(cse_files):d})"
            )

        # prepare the command line arguments for the python script and the slurm job file
        print(f"\nSubmitting: {struct_name:s}")
        cargs = f"--nnodes={nnodes:d} "
        cargs += f"--ncores={ncores_per_node:d} "
        cargs += f"--struct_path={struct_path:s} "
        cargs += f"--struct_name={struct_name:s} "
        cargs += f"--calc_path={current_calc_path:s} "
        cargs += f"--db_path={db_path:s}"
        if struct_name in finer_kgrid: # some materials converge badly
            adjusted_script_path = f"{base_path:s}/job_scripts/run_noctua_finer_kgrid.py"
        else:
            adjusted_script_path = script_path
        job_str = sbatch.job_template(
            adjusted_script_path,
            cargs,
            nnodes,
            ncores_per_node,
            memory_per_node=memory_per_node,
            wall_time=wall_time,
            job_name=job_name,
            queue=queue,
            conda_env=conda_env,
        )

        # go to the calculation directory and submit a job
        os.chdir(current_calc_path)
        with open("run.sh", "w") as f:
            f.write(job_str)
        os.system("sbatch run.sh")
        os.chdir(base_path)

        # iterate to the next structure
        idx += 1

    else:
        
        # wait
        print("All slots are filled, waiting for calculations to finish...\n", flush=True)
        time.sleep(300)

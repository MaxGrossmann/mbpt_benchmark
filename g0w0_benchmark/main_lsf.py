"""
DESCRIPTION:
    This script will initiate calculations for all materials in the benchmark dataset. The script checks
    every minute whether calculations can be started. If so, it starts them. The number of simultaneous 
    calculations is limited by the number of cores defined in the "control/ncores" file.  First, we will 
    start with materials that have one site in the unit cell. Then, we will move on to cells with two sites,
    and so on. If all the calculation slots are filled, the script checks every minute to see if a calculation
    has finished. If so, it starts a new one.
    
    Currently, the script is set up to run materials for our MBPT benchmark. Unfortunately, we cannot publish
    all of the input structures because they are from the ICSD. Nevertheless, in accordance with the ICSD license 
    agreement, we have provided a sample input file containing five materials. If you want to recalculate all the 
    materials in the benchmark, you will need to set up the input file yourself. To do so, download all the materials
    in the benchmark dataset from the ICSD, then check out the 'cif2input.ipynb' notebook in the top directory of the
    repository.
    
    To test the workflow, please use the 'main_local.py' script.
"""

# ignore all warnings
import warnings

warnings.filterwarnings("ignore")

# external imports
import os
import json
import time
import numpy as np
from copy import deepcopy

# local imports
import src.utils.basic_utils as basic_utils

# make sure we start in the right directory and get the base path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
base_path = os.getcwd()

"""
START USER INPUT
"""

# number of cores per job
ncores = 32

# requested job memory (does nothing when doing local calculation)
memory = 100000 # (mb)

# run only the crashed materials from all queues again
# the "queue" indicates the one queue which should be ignored
only_crashed_flag = False
queue = "BatchXL" # our cluster in Ilmenau has a "BatchXL" and "highmem" queue

# standard workflow order
workflows = [
    "qe_convergence_lda",
    "qe_convergence_pbe",
    "bandgap_convergence_lda",
    "bandgap_convergence_pbe",
    "yambo_g0w0_conv_lda",
    "yambo_g0w0_conv_pbe",
    "yambo_g0w0_ppa_lda",
    "yambo_g0w0_ppa_pbe",
]

# calculation directory with enough storage, i.e., a scratch directory
# THIS NEEDS TO BE AN ABSOLUTE PATH...
calc_dir = f"{base_path:s}/../qe_yambo_calc"

# name of the database directory (relative to the base directory)
# THIS PATH IS RELATIVE TO THIS SCRIPT...
database_dir = "./../qe_yambo_database"

# tag for the jobs
job_tag = "benchmark"

# LSF file name
lsf_name = "benchmark_job"

# name of the conda environment to use
conda_env = "mbpt_benchmark"

"""
END USER INPUT
"""

# check if the calculation directory exists
if not os.path.exists(os.path.join(calc_dir)):
    os.makedirs(os.path.join(calc_dir))

# same for the database directory
if not os.path.exists(os.path.join(base_path, database_dir)):
    os.makedirs(os.path.join(base_path, database_dir))

if (workflows[0] != "qe_convergence_lda") and (workflows[0] != "qe_convergence_pbe"):
    raise NameError(
        "Either 'qe_convergence_lda' or 'qe_convergence_pbe' must appear at the top of the workflow list!"
    )

# get the pickle file with all ICSD structures for the benchmark
input_file = os.path.join("input", "benchmark_structures.pkl")
if not os.path.exists(input_file):
    raise FileNotFoundError("The file 'benchmark_structures.pkl' does not exist, please run 'cif2input.ipynb' first!")

# run only crashed again or all
if only_crashed_flag == False:
    mat_pkl = basic_utils.icsd_load_pickle(input_file)
else:
    mat_pkl_all = basic_utils.icsd_load_pickle(input_file)
    df_crashed = basic_utils.crashed_jobs(calc_dir)
    crashed_ids = df_crashed[df_crashed["queue"] != queue]["id"].to_list()
    mat_pkl = []
    for mat in mat_pkl_all:
        if mat.parameters["id"] in crashed_ids:
            mat_pkl.append(mat)

# sort by number of sites
mat_pkl = basic_utils.icsd_sort_sites(mat_pkl)

# number of materials in the group
num_mats = len(mat_pkl)

# loop over all materials
curr_idx = 0
while curr_idx < num_mats:

    # number of sites in the unit cell
    num_sites = len(mat_pkl[curr_idx].structure)

    # current material id
    material_id = mat_pkl[curr_idx].parameters["id"]

    # check if a calculation directory for this material exists, if so skip it
    if os.path.exists(os.path.join(calc_dir, material_id)):
        print(f"The calculation directory for {material_id:s} already exists. Skipping this material.\n", flush=True)
        curr_idx += 1
        continue

    # check if database entry already exists
    wfs = deepcopy(workflows)
    if os.path.isfile(os.path.join(base_path, database_dir, f"{material_id:s}.json")):
        with open(os.path.join(base_path, database_dir, f"{material_id:s}.json")) as jfile:
            jdict = json.load(jfile)

        # check if qe_convergence_lda already done -> delete qe_convergence_lda from execution list
        if "qe_conv_lda" in jdict["parameters"]:
            wfs.remove("qe_convergence_lda")
            print(
                f"QE convergence with LDA pseudopotentials already was performed. Going ahead with {wfs[0]:s}\n",
                flush=True,
            )

        # check if qe_convergence_pbe already done -> delete qe_convergence_pbe from execution list
        if "qe_conv_pbe" in jdict["parameters"]:
            wfs.remove("qe_convergence_pbe")
            print(
                f"QE convergence with PBE pseudopotentials already was performed. Going ahead with {wfs[0]:s}\n",
                flush=True,
            )

        # check if bg_convergence_lda already done -> delete bg_convergence_lda from execution list
        if "bg_conv_lda" in jdict["parameters"]:
            wfs.remove("bandgap_convergence_lda")
            print(
                f"Band gap convergence with LDA pseudopotentials was already performed. Going ahead with the G0W0 convergence.\n",
                flush=True,
            )

        # check if bg_convergence_pbe already done -> delete bg_convergence_pbe from execution list
        if "bg_conv_pbe" in jdict["parameters"]:
            wfs.remove("bandgap_convergence_pbe")
            print(
                f"Band gap convergence with PBE pseudopotentials was already performed. Going ahead with the G0W0 convergence.\n",
                flush=True,
            )

        # check if yambo_g0w0_conv_pbe already done -> delete g0w0_convergence from execution list
        if "gw_conv_pbe" in jdict["parameters"]:
            wfs.remove("yambo_g0w0_conv_pbe")
            print(
                f"G0W0 convergence with PBE pseudopotentials was already performed. Going ahead with the final G0W0 calculation with PBE pseudopotentials.\n",
                flush=True,
            )

        # check if yambo_g0w0_conv_lda already done -> delete g0w0_convergence from execution list
        if "gw_conv_lda" in jdict["parameters"]:
            wfs.remove("yambo_g0w0_conv_lda")
            print(
                f"G0W0 convergence with LDA pseudopotentials was already performed. Going ahead with the final G0W0 calculation with LDA pseudopotentials.\n",
                flush=True,
            )

        # check if yambo_g0w0_ppa_pbe final calculation already done -> delete g0w0_final from execution list
        if "g0w0_ppa_pbe" in jdict["parameters"]:
            wfs.remove("yambo_g0w0_ppa_pbe")
            print(
                f"The final G0W0 calculation with PBE pseudopotentials was already performed.\n",
                flush=True,
            )

        # check if yambo_g0w0_ppa_lda final calculation already done -> delete g0w0_final from execution list
        if "g0w0_ppa_lda" in jdict["parameters"]:
            wfs.remove("yambo_g0w0_ppa_lda")
            print(
                f"The final G0W0 calculation with LDA pseudopotentials already performed.\n",
                flush=True,
            )

        if not wfs:
            curr_idx += 1
            print(
                f"Everything already done for {material_id:s}. Skip this material.\n",
                flush=True,
            )
            continue

    # check for control instructions
    if os.path.isfile("control/stop"):
        print("Stopping!")
        break
    if os.path.isfile("control/pause"):
        print("Paused!")
        time.sleep(60)
        continue

    # print the current calculation number and material
    with open("control/current_idx", "w") as f:
        f.write(
            f"{material_id:s}: {curr_idx + 1:d} of {num_mats:d} with {num_sites:d} sites\n"
        )

    # total number of cores we want/can use
    ncores_total = np.loadtxt("control/ncores")

    # find out how if any jobs can be started
    # (this is only only works for our IBM LSF job system...)
    ncores_curr = os.popen(f"bjobs -sum -J {job_tag}").read()
    ncores_curr = ncores_curr.split("\n")
    ncores_curr = ncores_curr[1].split()
    ncores_curr = sum(int(x) for x in ncores_curr)

    # check and print the current status
    max_jobs = int(ncores_total / ncores)
    curr_jobs = int(ncores_curr / ncores)
    print(
        f"Currently {curr_jobs:d} out of {max_jobs:d} possible jobs are running.\n",
        flush=True,
    )

    # start a calculation if possible
    if curr_jobs < max_jobs:
        basic_utils.start_calc_lsf(
            base_path,
            calc_dir,
            job_tag,
            lsf_name,
            database_dir,
            ncores,
            memory,
            material_id,
            mat_pkl[curr_idx],
            wfs,
            mat_pkl[curr_idx].parameters["lda_pw_cutoff_Ry"],
            queue,
            conda_env,
        )
        curr_idx += 1
    else:
        print(
            "All slots are filled, waiting for calculations to finish...\n",
            flush=True,
        )
        time.sleep(60)

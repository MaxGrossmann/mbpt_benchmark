"""
DESCRIPTION:
    This version is designed to be used interactively in a terminal for development and debugging purposes.
    You can choose from five example materials found in the directory './input/test/'.
"""

# ignore all warnings
import warnings

warnings.filterwarnings("ignore")

# external imports
import os
import json
import shutil
from copy import deepcopy
from pymatgen.entries.computed_entries import ComputedStructureEntry

# local imports
import src.utils.basic_utils as basic_utils

# make sure we start in the right directory and get the base path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
base_path = os.getcwd()

"""
START USER INPUT
"""

# material we want to calculate from the directory './input/test'
mat_name = "Si_icsd_51688_nsites_2"

# number of cores per job
ncores = 4

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
calc_dir = "/scratch/magr4985/g0w0_benchmark/calc_test"

# name of the database directory
# THIS PATH IS RELATIVE TO THIS SCRIPT...
database_dir = "./db_test"

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

# load the CSE from the JSON file of the selected material
mat_path = os.path.join(base_path, "input", "test", mat_name + ".json")
if not os.path.exists(mat_path):
    raise FileNotFoundError(f"No JSON file exists for '{mat_name:s}'. Please run 'cif2input.ipynb' first!")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mat = ComputedStructureEntry.from_dict(json.load(open(mat_path)))

# current material id
material_id = mat.parameters["id"]

# check if a calculation directory for this material exists, if so skip it
if os.path.exists(os.path.join(calc_dir, material_id)):
    print(f"The calculation directory for {material_id:s} already exists. It is being removed so that we can restart.\n", flush=True)
    shutil.rmtree(os.path.join(calc_dir, material_id))
if os.path.isfile(os.path.join(base_path, database_dir, f"{material_id:s}.json")):
    print(f"The database entry for {material_id:s} already exists. It is being removed so that we can restart.\n",flush=True)
    os.remove(os.path.join(base_path, database_dir, f"{material_id:s}.json"))

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

basic_utils.start_calc_local(
    base_path,
    calc_dir,
    database_dir,
    ncores,
    material_id,
    mat,
    wfs,
    mat.parameters["lda_pw_cutoff_Ry"],
    conda_env,
)
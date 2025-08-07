"""
This script starts the chosen workflows from main.py.
"""

# external imports
import sys
import json
import warnings
from pymatgen.entries.computed_entries import ComputedStructureEntry

# local imports
import src.utils.basic_utils as basic_utils
from src.workflows.qe_convergence_pbe import qe_convergence_pbe
from src.workflows.bandgap_convergence_pbe import bandgap_convergence_pbe
from src.workflows.qe_convergence_lda import qe_convergence_lda
from src.workflows.bandgap_convergence_lda import bandgap_convergence_lda
from src.workflows.yambo_g0w0_conv_pbe import yambo_g0w0_conv_pbe
from src.workflows.yambo_g0w0_conv_lda import yambo_g0w0_conv_lda
from src.workflows.yambo_g0w0_ppa_pbe import yambo_g0w0_ppa_pbe
from src.workflows.yambo_g0w0_ppa_lda import yambo_g0w0_ppa_lda

# base directory
base_dir = str(sys.argv[1])

# number of cores
ncores = int(sys.argv[2])

# get the id from the args that are called with this script
id = str(sys.argv[3])

# database directory path
db_entry_path = str(sys.argv[4])

# get list of workflows (bad code...)
workflows = list(sys.argv[5].strip("[]").replace("'", "").replace(" ", "").split(","))

# calculation directory
calc_dir = str(sys.argv[6])

# local flag: computing on hpc or locally
try:
    local_flag = bool(sys.argv[7])
except:
    local_flag = False

# get the current job id (0 if the script is running locally)
if not local_flag:
    job_id = basic_utils.get_job_id()

# load the database entry
with open(db_entry_path, "r") as json_file:
    json_dict = json.load(json_file)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cse = ComputedStructureEntry.from_dict(json_dict)

# dictionary connecting the workflow name to the function
# (update this dictionary if you add a new workflow)
workflow_dict = {
    "qe_convergence_pbe": qe_convergence_pbe,
    "qe_convergence_lda": qe_convergence_lda,
    "bandgap_convergence_pbe": bandgap_convergence_pbe,
    "bandgap_convergence_lda": bandgap_convergence_lda,
    "yambo_g0w0_conv_pbe": yambo_g0w0_conv_pbe,
    "yambo_g0w0_conv_lda": yambo_g0w0_conv_lda,
    "yambo_g0w0_ppa_pbe": yambo_g0w0_ppa_pbe,
    "yambo_g0w0_ppa_lda": yambo_g0w0_ppa_lda,
}

# loop over all workflows, update the maximum memory used and the database entry
for wf in workflows:
    # run the workflow
    if wf not in workflow_dict:
        sys.exit(f"Workflow {wf:s} not found in the workflow dictionary!")
    cse = workflow_dict[wf](cse, base_dir, ncores, calc_dir, db_entry_path)

    # update the required maximum memory
    if not local_flag:
        if job_id > 0:
            max_mem = basic_utils.get_max_mem(job_id)
            basic_utils.update_if_larger(cse.parameters, "max_mem", max_mem)

    # update the database entry
    with open(db_entry_path, "w") as f:
        json_dict = cse.as_dict()
        json.dump(json_dict, f, cls=basic_utils.NumpyEncoder)

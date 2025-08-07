"""
DESCRIPTION:
    This version is designed to be used interactively in a terminal for development and debugging purposes.
    Running a material again will clean up the calculation directory. If a database entry already exists for 
    a material and the convergence thresholds have not changed, the convergence routines will be skipped. If
    a database entry exists and the convergence thresholds have changed, everything will be recalculated from 
    scratch. This also happens if the number of cores changes, as the timing information in the database would 
    be incorrect.
    
    The convergence parameters used here are more lenient than those used in the benchmark. This is because this
    script was primarily used for development and testing purposes.  The convergence parameters used for the
    benchmark are located in the scripts within the 'job_scripts/' directory.
    
    You can choose from five example materials found in the directory './structures/test/' (BeTe, C, Ne, Si, and TlBr).
"""

# external imports
import os
import sys

# internal imports
from qsgw_workflow.utils.system_config import set_execution_mode
from qsgw_workflow.workflows import SemiconductorWorkflow

# make sure we start in the right directory and get the base path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
base_path = os.getcwd()

"""
START OF USER INPUT.
"""

# name of the CSE file for which we want to run the workflow structure (without file ending)
# -> see /structures/test for some examples
struct_name = "BeTe" # BeTe is a good test material because the calculation converges very quickly

# number of cores
ncores = 4

# path to a directory with CSE files (absolute path)
struct_path = f"{base_path:s}/structures/test"

# path to a directory where all calculations will take place, e.g., a scratch directory (absolute path)
calc_path = f"{base_path:s}/calc_test"

# path to a directory where the database file will be saved (absolute path)
db_path = f"{base_path:s}/db_test"

# starting k-grid densities for the k-grid convergence (dft_kppa > qsgw_kppa)
dft_kppa = 1000
qsgw_kppa = 100 # (lower starting point so we can easily test materials on a laptop)

# convergence thresholds
dft_tol = 1e-4   # DFT convergence threshold for the k-grid and 'gmax' (Ry)
eps_tol = 0.95   # dielectric tensor k-grid convergence threshold(SC < 1)
qsgw_tol = 0.01  # QSGW k-grid convergence threshold (Ry) (larger threshold so we can easily test materials on a laptop)

# flag to remove database entries for the selected material
# set this to 'True' when performing normal computations or when debugging the code
# set this to 'False' when debugging calculation restarts
rm_db = False

"""
END OF USER SECTION.
"""

# adjust how 'os.system' and 'mpirun' calls are done
set_execution_mode("local")

# sanity check
if not os.path.exists(os.path.join(struct_path, struct_name + ".json")):
    raise FileNotFoundError(f"The material '{struct_name:s}' does not exist in the directory '{struct_path:s}'. Please run 'cif2input.ipynb' first!")

# create the calculation directory and go to the calculation directory
calc_path = f"{calc_path}/{struct_name:s}"
os.makedirs(calc_path, exist_ok=True)
os.chdir(calc_path)
    
# database directory setup 
if not os.path.exists(db_path):
    os.makedirs(db_path)
if rm_db: # delete it if it exists and if 'rm_db' is set to 'True'
    db_entry = os.path.join(db_path, f"{struct_name:s}.json")
    if os.path.exists(db_entry):
        os.remove(db_entry)

# create and run the workflow
wf = SemiconductorWorkflow(
    calc_path, 
    db_path, 
    struct_path,
    struct_name, 
    1, # number of nodes
    ncores, 
    dft_kppa=dft_kppa,
    qsgw_kppa=qsgw_kppa,
    dft_tol=dft_tol, 
    eps_tol=eps_tol, 
    qsgw_tol=qsgw_tol
)
wf.run()
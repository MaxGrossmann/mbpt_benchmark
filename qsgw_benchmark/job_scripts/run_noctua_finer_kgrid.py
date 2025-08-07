"""
Only used for some materials, e.g., BaO, CsAu...
"""

# external imports
import json
import argparse

# internal imports
from qsgw_workflow.utils.system_config import set_execution_mode
from qsgw_workflow.workflows import SemiconductorWorkflow

"""
DESCRIPTION.
    This script is optimized to run in a SBATCH job on Noctua 2 (Paderborn Center for Parallel Computing - PC2, Germany).
    If a database entry exists for the material and the convergence thresholds have not changed, the convergence routines will be skipped.
    If the database entry exists and the convergence thresholds have changed, everything will be recalculated from scratch.
    See 'main_noctua.py' for details.
"""

# parse all variables from the command line
parser = argparse.ArgumentParser(description="QSGW WORKFLOW.")
parser.add_argument("--nnodes", type=int, required=True, help="Number of nodes.")
parser.add_argument("--ncores_per_node", type=int, required=True, help="Number of cores per node.")
parser.add_argument("--struct_path", type=str, required=True, help="Path to a directory with CSE files (absolute path).")
parser.add_argument("--struct_name", type=str, required=True, help="Name of the CSE file for which we want to run the workflow structure (without file ending).")
parser.add_argument("--calc_path", type=str, required=True, help="Path to a directory where all calculations will take place, e.g., a scratch directory (absolute path).")
parser.add_argument("--db_path", type=str, default=True, help="Path to a directory where the database file will be saved (absolute path).")
args = parser.parse_args()

# convert args to a dictionary and print them for logging purposes
args_dict = vars(args)
print("\nParsed Arguments:")
print(json.dumps(args_dict, indent=4))

# starting k-grid densities for the k-grid convergence
# (THE VALUES HERE ARE HIGHER...)
dft_kppa = 1500
qsgw_kppa = 1000

# convergence thresholds
dft_tol = 1e-4     # DFT convergence threshold for the k-grid and 'gmax' (Ry)
eps_tol = 0.95     # dielectric tensor k-grid convergence threshold (SC)
qsgw_tol = 0.00184 # QSGW k-grid convergence threshold (Ry) -> ~25 meV, i.e., benchmark settings

# adjust how 'os.system' and 'mpirun' calls are done
set_execution_mode("noctua")

# create and run the workflow
wf = SemiconductorWorkflow(
    args.calc_path, 
    args.db_path, 
    args.struct_path,
    args.struct_name, 
    args.nnodes,
    args.ncores_per_node, 
    dft_kppa=dft_kppa,
    qsgw_kppa=qsgw_kppa,
    dft_tol=dft_tol, 
    eps_tol=eps_tol, 
    qsgw_tol=qsgw_tol,
)
wf.run()
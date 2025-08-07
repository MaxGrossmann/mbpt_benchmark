"""
This workflow runs a simple convergence algorithm to converge
the band gap with respect to the cutoff and k-point grid for a 
given material using PBE pseudopotentials.
"""

# external imports
import os
import time

# local imports
import src.utils.qe_helper as qe_helper
import src.utils.qe_write as qe_write
import src.utils.qe_runner as qe_runner
from src.utils.calc_data_class import calc_data

"""
Main workflow for the Quantum ESPRESSO band gap convergence.
"""

def bandgap_convergence_pbe(cse, base_dir, ncores, calc_dir, db_entry_path):

    # log message
    print(f"\nStarting: {os.path.basename(__file__)}", flush=True)

    # get some variables from the database entry
    id = cse.parameters["id"]
    structure = cse.structure
    ibrav = cse.parameters["ibrav"]

    # create and go into calculation dirrectory
    if not os.path.exists(os.path.join(calc_dir, id, "pw_pbe")):
        os.makedirs(os.path.join(calc_dir, id, "pw_pbe"))
    os.chdir(os.path.join(calc_dir, id, "pw_pbe"))

    # initialize the start time and the parameter dictionary
    start_time = time.time()
    param_dict = {}

    # some sensible hyperparameters for high-throughput calculation
    maxiter = 20
    delta_kppa = 1500
    delta_cutoff = 5
    conv_thresh = 0.025 # threshold for band gap convergence, Borlido et al. [https://doi.org/10.1021/acs.jctc.9b00322] used 50 meV

    # update the parameter dictionary
    param_dict.update(
        {
            "bg_conv_maxiter": maxiter,
            "bg_delta_kppa": delta_kppa,
            "bg_delta_cutoff": delta_cutoff,
            "bg_conv_thresh": conv_thresh,
        }
    )

    # extract the convergence parameters from 'qe_convergence_pbe'
    pw_cutoff = cse.parameters["qe_conv_pbe"]["qe_conv_steps"][-1][0]
    kppa = cse.parameters["qe_conv_pbe"]["qe_conv_steps"][-1][1]

    # pseudopotential directory
    pseudo = os.path.join(base_dir, "pseudo/PBE")

    # setup an nscf calculation 
    calc_data_curr = calc_data(
        structure,
        id=id,
        ibrav=ibrav,
        calc_type="nscf",
        pw_cutoff=pw_cutoff,
        kppa=kppa,
        pseudo=pseudo,
    )

    # collect file names
    fn = []

    # run the inital calculation
    filename = qe_runner.qe_pw_run(
        calc_data_curr, qe_write.write_pw, ncores, kwargs={"disk_io": "high"}
    )
    fn.append(filename)

    # list of convergence parameters (cutoff, kppa, total energy)
    curr_energy = qe_helper.bg_get_bandgap(calc_data_curr)
    bg_conv_steps = [[pw_cutoff, kppa, curr_energy]]

    # log message
    print(
        f"band gap convergence threshold = {conv_thresh:.4f} eV",
        flush=True,
    )
    print("cutoff (Ry)  kppa   band gap (eV)", flush=True)
    print(
        f"{bg_conv_steps[-1][0]:<11d}  {bg_conv_steps[-1][1]:<5d}  {bg_conv_steps[-1][2]:.6f}",
        flush=True,
    )

    # k-point convergence
    iter = 0

    # check whether any k-points were actually added, if not, increase kppa again
    calc_data0 = calc_data_curr
    while calc_data_curr.k_points_grid == calc_data0.k_points_grid:
        calc_data0 = calc_data_curr
        kppa += delta_kppa
        calc_data_curr = calc_data(
            structure,
            id=id,
            ibrav=ibrav,
            pw_cutoff=pw_cutoff,
            kppa=kppa,
            pseudo=pseudo,
            degauss=calc_data_curr.degauss,
        )

    # run calculation with finer k-point grid
    filename = qe_runner.qe_pw_run(
        calc_data_curr, qe_write.write_pw, ncores, kwargs={"disk_io": "low"}
    )
    fn.append(filename)
    iter += 1

    # read out energy from output file, print it and add it to the energies list
    curr_energy = qe_helper.bg_get_bandgap(calc_data_curr)
    bg_conv_steps.append([pw_cutoff, kppa, curr_energy])

    # log message
    print(
        f"{bg_conv_steps[-1][0]:<11d}  {bg_conv_steps[-1][1]:<5d}  {bg_conv_steps[-1][2]:.6f}",
        flush=True,
    )

    # compare band gap energies between coarse and finer k-point grid
    # while energy difference is above threshold, make k-point grid finer, repeat calculation etc.
    while abs(bg_conv_steps[-1][-1] - bg_conv_steps[-2][-1]) > conv_thresh:
        calc_data0 = calc_data_curr
        # check whether any k-points were actually added
        while calc_data_curr.k_points_grid == calc_data0.k_points_grid:
            calc_data0 = calc_data_curr
            kppa += delta_kppa
            calc_data_curr = calc_data(
                structure,
                id=id,
                ibrav=ibrav,
                calc_type="nscf",
                pw_cutoff=pw_cutoff,
                kppa=kppa,
                pseudo=pseudo,
                degauss=calc_data_curr.degauss,
            )

        # start calculation with finer k-point grid
        filename = qe_runner.qe_pw_run(
            calc_data_curr, qe_write.write_pw, ncores, kwargs={"disk_io": "high"}
        )
        fn.append(filename)
        iter += 1

        # read out energy from output file, print it and add it to the energies list
        curr_energy = qe_helper.bg_get_bandgap(calc_data_curr)
        bg_conv_steps.append([pw_cutoff, kppa, curr_energy])

        # log message
        print(
            f"{bg_conv_steps[-1][0]:<11d}  {bg_conv_steps[-1][1]:<5d}  {bg_conv_steps[-1][2]:.6f}",
            flush=True,
        )

        # safety feature
        if iter >= maxiter:
            print(
                f"k-point density not converged after {maxiter} iterations.\n",
                flush=True,
            )
            raise Exception("k-point density not converged...\n")

    # log message
    print("The band gap converged with respect to the cutoff and k-point grid.", flush=True)

    # calculation time for the QE convergence
    bg_conv_time = time.time() - start_time

    # save the converged parameters and steps in the parameter dictionary
    param_dict.update(
        {
            "bg_conv_steps": bg_conv_steps,
            "bg_conv_time": bg_conv_time,
        }
    )

    # keep the final input/output files in case one wants to check them
    # but delete all other files
    for f in fn[:-1]:
        os.remove(f + ".in")
        os.remove(f + ".out")

    # delete all wavefuction files, but keep density file
    for file in os.listdir(os.path.join("out", f"{id}.save")):
        if "wfc" in file:
            os.remove(os.path.join("out", f"{id}.save", file))
    for file in os.listdir("out"):
        if "wfc" in file:
            os.remove(os.path.join("out", file))
    for file in os.listdir():
        if "wfc" in file:
            os.remove(file)

    # update the parameters in the database entry
    cse.parameters.update(
        {"bg_conv_pbe": param_dict, "indirect_gap_pbe": bg_conv_steps[-1][2]}
    )

    # go back to main calculation directory
    os.chdir("..")

    # log message
    print("Convergence files cleaned up.", flush=True)

    return cse

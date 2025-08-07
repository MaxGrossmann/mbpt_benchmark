"""
This workflow runs a simple convergence algorithm to converge
the plane-wave cutoff and the number of k-points for a given material using LDA pseudopotentials.
The convergence is in regards to the total energy per atom in the unit cell.
"""

# external imports
import os
import time

# local imports
import src.utils.qe_helper as qe_helper
import src.utils.qe_write as qe_write
import src.utils.qe_runner as qe_runner
from src.utils.calc_data_class import calc_data
from src.utils.unit_conversion import ev2ha

"""
Main workflow for the Quantum ESPRESSO k-grid and cutoff convergence.
"""

def qe_convergence_lda(cse, base_dir, ncores, calc_dir, db_entry_path):

    # log message
    print(f"\nStarting: {os.path.basename(__file__)}", flush=True)

    # get some variables from the database entry
    id = cse.parameters["id"]
    structure = cse.structure
    ibrav = cse.parameters["ibrav"]
    pw_cutoff = cse.parameters["lda_pw_cutoff_Ry"]

    # create and go into calculation dirrectory
    if not os.path.exists(os.path.join(calc_dir, id, "pw_lda")):
        os.makedirs(os.path.join(calc_dir, id, "pw_lda"))
    os.chdir(os.path.join(calc_dir, id, "pw_lda"))

    # initialize the start time and the parameter dictionary
    start_time = time.time()
    param_dict = {}

    # print the composition of the structure
    print(f"\n{id:s} [{structure.composition}]\n", flush=True)

    # log message
    print("Starting QE convergence with LDA pseudopotential:", flush=True)

    # some sensible hyperparameters for high-throughput calculation
    maxiter = 20
    delta_kppa = 1500
    delta_cutoff = 5
    conv_thresh = 0.001 # eV/atom
    conv_thresh = ev2ha(conv_thresh)
    conv_thresh = conv_thresh * len(structure.sites)

    # update the parameter dictionary
    param_dict.update(
        {
            "qe_conv_maxiter": maxiter,
            "qe_delta_kppa": delta_kppa,
            "qe_delta_cutoff": delta_cutoff,
            "qe_conv_thresh": conv_thresh,
        }
    )

    # pseudopotential directory
    pseudo = os.path.join(base_dir, "pseudo/LDA")

    # create dummy scf calculation (useful to check k-grid later)
    calc_data0 = calc_data(  # this is a dummy calculation need to check
        structure,
        id=id,
        ibrav=ibrav,
        calc_type="scf",
        pw_cutoff=pw_cutoff,
        kppa=1500,
        pseudo=pseudo,
    )

    # extract the initial parameters and create a calculation which is changed later on
    structure = calc_data0.structure
    id = calc_data0.id
    ibrav = calc_data0.ibrav
    pw_cutoff = calc_data0.pw_cutoff
    kppa = calc_data0.kppa
    pseudo = calc_data0.pseudo
    calc_data_curr = calc_data(
        structure,
        id=id,
        ibrav=ibrav,
        pw_cutoff=pw_cutoff,
        kppa=kppa,
        pseudo=pseudo,
    )

    # collect file names
    fn = []

    # run the inital calculation
    filename = qe_runner.qe_pw_run(
        calc_data_curr,
        qe_write.write_pw,
        ncores,
        kwargs={"input_dft": "lda", "disk_io": "low"},
    )
    fn.append(filename)

    # list of convergence parameters (cutoff, kppa, total energy)
    curr_energy = qe_helper.qe_read_tot_energy(calc_data_curr)
    qe_conv_steps = [[pw_cutoff, kppa, curr_energy]]

    # log message
    print(
        f"convergence threshold = {conv_thresh:.6f} Ha/atom", flush=True
    )
    print("cutoff (Ry)  kppa   total energy (Ha)", flush=True)
    print(
        f"{qe_conv_steps[-1][0]:<11d}  {qe_conv_steps[-1][1]:<5d}  {qe_conv_steps[-1][2]:.6f}",
        flush=True,
    )

    # cutoff convergence
    iter = 0

    # increase cutoff, run calculation, compare with initial calculation
    pw_cutoff += delta_cutoff
    calc_data_curr = calc_data(
        structure,
        id=id,
        ibrav=ibrav,
        pw_cutoff=pw_cutoff,
        kppa=kppa,
        pseudo=pseudo,
        degauss=calc_data_curr.degauss,
    )

    # run calculation with increased cutoff energy
    filename = qe_runner.qe_pw_run(
        calc_data_curr,
        qe_write.write_pw,
        ncores,
        kwargs={"input_dft": "lda", "disk_io": "low"},
    )
    fn.append(filename)
    iter += 1

    # read out energy from output file, print it and add it to the energies list
    curr_energy = qe_helper.qe_read_tot_energy(calc_data_curr)
    qe_conv_steps.append([pw_cutoff, kppa, curr_energy])

    # log message
    print(
        f"{qe_conv_steps[-1][0]:<11d}  {qe_conv_steps[-1][1]:<5d}  {qe_conv_steps[-1][2]:.6f}",
        flush=True,
    )

    # while the energy difference is below the convergence threshold, repeat the above steps
    while abs(qe_conv_steps[-1][-1] - qe_conv_steps[-2][-1]) > conv_thresh:
        # increase the cutoff energy
        pw_cutoff += delta_cutoff
        calc_data_curr = calc_data(
            structure,
            id=id,
            ibrav=ibrav,
            pw_cutoff=pw_cutoff,
            kppa=kppa,
            pseudo=pseudo,
            degauss=calc_data_curr.degauss,
        )

        # run calculation with higher cutoff energy
        filename = qe_runner.qe_pw_run(
            calc_data_curr,
            qe_write.write_pw,
            ncores,
            kwargs={"input_dft": "lda", "disk_io": "low"},
        )
        fn.append(filename)
        iter += 1

        # read out energy from output file, print it and add it to the energies list
        curr_energy = qe_helper.qe_read_tot_energy(calc_data_curr)
        qe_conv_steps.append([pw_cutoff, kppa, curr_energy])

        # log message
        print(
            f"{qe_conv_steps[-1][0]:<11d}  {qe_conv_steps[-1][1]:<5d}  {qe_conv_steps[-1][2]:.6f}",
            flush=True,
        )

        # safety feature
        if iter >= maxiter:
            print(
                f"Cutoff not converged after {maxiter} iterations. Proceeding to k-point convergence...\n",
                flush=True,
            )
            raise Exception("cutoff energy not converged...\n")

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
        calc_data_curr,
        qe_write.write_pw,
        ncores,
        kwargs={"input_dft": "lda", "disk_io": "low"},
    )
    fn.append(filename)
    iter += 1

    # read out energy from output file, print it and add it to the energies list
    curr_energy = qe_helper.qe_read_tot_energy(calc_data_curr)
    qe_conv_steps.append([pw_cutoff, kppa, curr_energy])

    # read out number of electrons
    num_elec = qe_helper.read_num_electrons(calc_data_curr)

    # log message
    print(
        f"{qe_conv_steps[-1][0]:<11d}  {qe_conv_steps[-1][1]:<5d}  {qe_conv_steps[-1][2]:.6f}",
        flush=True,
    )

    # compare energies between k-point grids
    # while energy difference is above threshold, make k-point grid finer, repeat calculation etc.
    while abs(qe_conv_steps[-1][-1] - qe_conv_steps[-2][-1]) > conv_thresh:
        calc_data0 = calc_data_curr
        # check whether any k-points were actually added
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

        # start calculation with finer k-point grid
        filename = qe_runner.qe_pw_run(
            calc_data_curr,
            qe_write.write_pw,
            ncores,
            kwargs={"input_dft": "lda", "disk_io": "low"},
        )
        fn.append(filename)
        iter += 1

        # read out energy from output file, print it and add it to the energies list
        curr_energy = qe_helper.qe_read_tot_energy(calc_data_curr)
        qe_conv_steps.append([pw_cutoff, kppa, curr_energy])

        # log message
        print(
            f"{qe_conv_steps[-1][0]:<11d}  {qe_conv_steps[-1][1]:<5d}  {qe_conv_steps[-1][2]:.6f}",
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
    print("Convergence reached.", flush=True)

    # calculation time for the QE convergence
    qe_conv_time = time.time() - start_time

    # save the converged parameters and steps in the parameter dictionary
    param_dict.update(
        {
            "qe_conv_steps": qe_conv_steps,
            "qe_conv_time": qe_conv_time,
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
    cse.parameters.update({"qe_conv_lda": param_dict, "num_elec_lda": num_elec})

    # go back to main calculation directory
    os.chdir("..")

    # log message
    print("Convergence files cleaned up.", flush=True)

    return cse

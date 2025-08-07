"""
This workflow efficiently finds the convergence parameters for a GW calculation 
starting from an LDA starting point. First, the convergence parameters for W are
found using the coordinate search algorithm on a Gamma-only k-point grid. Then, 
the k-point grid density is increased and one GW calculation per k-point grid is 
performed using small parameters in W, i.e. (N_b=400, G_cut=8), until the gap converges with
respect to the k-point grid. In some cases, this may result in an underconverged calculation.
If high accuracy is required, check the final parameters again. For details read:
https://doi.org/10.1038/s41524-024-01311-9
"""

# external imports
import os
import time
import json
import shutil
import warnings
import numpy as np
from pymatgen.entries.computed_entries import ComputedStructureEntry

# local imports
import src.utils.qe_write as qe_write
import src.utils.qe_helper as qe_helper
import src.utils.qe_runner as qe_runner
import src.utils.yambo_runner as yambo_runner
import src.utils.yambo_helper as yambo_helper
import src.utils.yambo_write as yambo_write
from src.utils.calc_data_class import calc_data
from src.utils.basic_utils import get_kpt_grid

def yambo_g0w0_conv_lda(cse, base_dir, ncores, calc_dir, db_entry_path):

    # material id out of cse
    id = cse.parameters["id"]

    # print the workflow name
    print(f"\nStarting: {os.path.basename(__file__)}", flush=True)

    # workflow directory
    wf_dir = os.path.join(os.getcwd(), "yambo_g0w0_conv_lda")
    if os.path.exists(wf_dir):
        shutil.rmtree(wf_dir)
    if not os.path.exists(wf_dir):
        os.mkdir(wf_dir)
    os.chdir(wf_dir)

    # load some database entries
    json_path = db_entry_path
    with open(json_path, "r") as json_file:
        json_dict = json.load(json_file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cse = ComputedStructureEntry.from_dict(json_dict)

    # number of electrons
    num_elec = cse.parameters["num_elec_lda"]

    # get converged parameters
    structure = cse.structure
    ibrav = cse.parameters["ibrav"]
    pw_cutoff_qe = cse.parameters["qe_conv_lda"]["qe_conv_steps"][-1][0] # dummy value, get overwrittens in 'yambo_run_gw_conv_cs'
    kppa_qe = cse.parameters["qe_conv_lda"]["qe_conv_steps"][-1][1] # used for initial scf in 'yambo_run_gw_conv_cs' 

    # pseudopotential directory
    pseudo = os.path.join(base_dir, "pseudo/LDA")

    # create calc_data from converged parameters
    calc_data_conv = calc_data(
        structure=structure,
        id=id,
        ibrav=ibrav,
        calc_type="scf",
        pw_cutoff=pw_cutoff_qe,
        kppa=kppa_qe,
        pseudo=pseudo,
    )

    # copy out directory from QE and band gap convergence to Yambo directory
    shutil.copytree("../pw_lda/out", "out")

    # good starting value for the maximum number of bands in the nscf
    # this will be increased if the convergence fails
    n_bands = 1000

    # convergence threshold
    conv_thr = 0.025 # eV

    # starting parameter and parameter step size
    bnd_start = 200
    bnd_step = 100
    cut_start = 4
    cut_step = 4

    # create a dictionary with all important workflows parameters and results
    gw_dict = {
        "gw_conv_thr": conv_thr,
        "gw_bnd_start": bnd_start,
        "gw_bnd_step": bnd_step,
        "gw_cut_start": cut_start,
        "gw_cut_step": cut_step,
    }

    # list to store parameters and associated calculation results
    gw_list = []

    """
    Gamma-only convergence of the parameters in W.
    """

    # start the timing
    start_time = time.time()

    # change the number of cores for small k-point grid calculations
    # as the parallel structure will not work well with too many cores
    # and too few k-points (same applies to nscf calculations)
    if num_elec < 5:
        ncpu = np.min([4, ncores])
    else:
        ncpu = np.min([8, ncores])

    # create the output files
    with open(f"{id}_gw_conv.txt", "w+") as f:
        f.write(
            f"kppa \t kpt \t    nk_ir  bnd_x  bnd_g  cutsrc  gap      #GW  time (s)  ncpu\n"
        )

    # converge the parameters in W doing a gamma-only GW calculation
    gw = None
    ngw = 0
    while True:
        (
            conv_flag,
            bnd_increase_flag,
            gw,
            filename_nscf,
            pw_cutoff,
        ) = yambo_runner.yambo_run_gw_conv_cs(
            calc_data_conv,
            n_bands,
            0, # gamma-only convergence of W parameters
            gw,
            base_dir,
            ncpu,
            bnd_start=bnd_start,
            bnd_step=bnd_step,
            cut_start=cut_start,
            cut_step=cut_step,
            conv_thr=conv_thr,
            input_dft="lda",
        )

        ngw += len(gw.fn)
        if conv_flag:
            print(
                f"\nThe CS algorithm finished successfully after {len(gw.fn)} GW calculation.\n",
                flush=True,
            )
            break
        elif not conv_flag and bnd_increase_flag:
            print("Increasing number of bands in the nscf step...", flush=True)
            os.remove(f"{filename_nscf}.in")
            os.remove(f"{filename_nscf}.out")
            shutil.rmtree("out")
            n_bands += int(2.5 * bnd_step)
            if n_bands >= 3000:
                break
        elif not conv_flag and not bnd_increase_flag:
            break

    # append the output file
    kstr = f"{1:d}x{1:d}x{1:d}"
    with open(f"{id}_gw_conv.txt", "a+") as f:
        if n_bands >= 3000:
            f.write("CONVERGENCE FAILED. TOO MANY BANDS WERE REQUESTED")
            open("gw_gamma_too_many_bands.txt", "w").close()
            raise Exception("Convergence failed.")
        elif not conv_flag and not bnd_increase_flag:
            f.write("CONVERGENCE FAILED. TOO MANY BANDS WERE REQUESTED.")
            open("gw_gamma_conv_failed.txt", "w").close()
            raise Exception("Convergence failed.")
        else:
            f.write(f"{0:<4d} \t {kstr:<9}  {gw.num_kpt:<5d}  ")
            f.write(
                f"{int(gw.final_point[0]):<5d}  {int(gw.final_point[0]):<5d}  {int(gw.final_point[1]):<6d}  "
                + f"{gw.final_point[2]:<2.5f}  {ngw:<3d}  "
            )
        f.write(f"{int(np.ceil((time.time() - start_time))):<8d}  {ncpu:<4d}\n")

        # append the output with the results from the starting point calculation as a reference for the k-point grid convergence
        f.write(f"{0:<4d} \t {kstr:<9}  {gw.num_kpt:<5d}  ")
        f.write(
            f"{int(gw.grid[0][0]):<5d}  {int(gw.grid[0][0]):<5d}  {int(gw.grid[0][1]):<6d}  "
            + f"{gw.grid[0][2]:<2.5f}  {1:<3d}  "
        )
        f.write(f"{int(np.ceil(gw.grid_time[0])):<8d}  {ncpu:<4d}\n")

    # append the gw dictionary
    gw_dict.update(
        {
            "kpt_bnd_idx": gw.kpt_bnd_idx,
            "gw_bands": int(gw.grid[0][0]),
            "gw_cutoff": int(gw.grid[0][1]),
            "w_conv_grid": gw.grid,
            "pwcutoff": pw_cutoff,
            "w_conv_time": gw.grid_time,
            "w_ngw": ngw,
            "w_z_factor": gw.z_factor,
        }
    )

    # append the results to the list
    gw_list.append(
        [
            {"kppa": 0}, # kppa
            {"kpt_grid": kstr}, # kpt grid
            {"num_kpts": gw.num_kpt}, # number of irreducible k-points
            {"X_bnds": gw.final_point[0]}, # number of bands in X
            {"G_bnds": gw.final_point[0]}, # number of bands in G
            {"X_cutoff": gw.final_point[1]}, # cutoff for X (Ry)
            {"gamma_gap": gw.final_point[2]}, # gap at the Gamma point in eV
            {"num_gw_calc": ngw}, # number of GW calculations
            {
                "wall_time": int(np.ceil((time.time() - start_time)))
            }, # wall time in seconds
            {"num_cores": ncpu}, # number of cores
            {"z-factor": "None"}, # z-factor
        ]
    )

    # append the results from the starting point as a reference for the k-point grid convergence to the list
    gw_list.append(
        [
            {"kppa": 0}, # kppa
            {"kpt_grid": kstr}, # kpt grid
            {"num_kpts": gw.num_kpt}, # number of irreducible k-points
            {"X_bnds": gw.grid[0][0]}, # number of bands in X
            {"G_bnds": gw.grid[0][0]}, # number of bands in G
            {"X_cutoff": gw.grid[1][1]}, # cutoff for X (Ry)
            {"gamma_gap": gw.grid[2][1]}, # gap at the Gamma point in eV
            {"num_gw_calc": 1}, # number of GW calculations
            {"wall_time": int(np.ceil(gw.grid_time[0]))}, # wall time in seconds
            {"num_cores": ncpu}, # number of cores
            {"z-factor": "None"}, # z-factor
        ]
    )

    """
    k-point grid convergence with low parameters in W.
    """

    # now converge the k-point grid using the cheap starting values from W
    kppa = [0]
    k_grid = [np.array(get_kpt_grid(structure, kppa[0]))]
    kgrid_gap = [gw.grid[0][2]]
    diff_gap = 1
    iter = 1
    max_iter = 8
    kppa_step = 10

    # log messages
    print(
        f"Running the k-point grid convergence ({bnd_start:d} bands, {cut_start:d} Ry):",
        flush=True,
    )
    kstr = "1x1x1"
    print(
        f"k-point grid: {kstr:>8s} "
        + f"(kppa = {0:>5d}) "
        + f"Gap = {kgrid_gap[0]:6f} eV",
        flush=True,
    )

    while diff_gap > conv_thr:
        # start the timing
        start_time_kpt = time.time()

        # increase the k-point grid density
        kppa.append(kppa[-1] + kppa_step)

        # change the number of cores for small k-point grid calculations
        # as the parallel structure is not working well with too many cores
        # (also the nscf has convergence problems)
        if kppa[iter] < 50:
            if num_elec < 5:
                ncpu = np.min([4, ncores])
            else:
                ncpu = np.min([8, ncores])
        else:
            ncpu = ncores

        # obtain the new k-point grid for the nscf step
        k_grid.append(np.array(get_kpt_grid(structure, kppa[iter])))

        # increase the k-point density until the k-point grid changes
        while np.sum(np.abs(k_grid[iter] - k_grid[iter - 1])) == 0:
            kppa[iter] += kppa_step
            k_grid[iter] = np.array(get_kpt_grid(structure, kppa[iter]))

        # create nscf calculation based on scf calculation and run it
        calc_data_nscf = calc_data(
            structure,
            id=id,
            ibrav=ibrav,
            calc_type="nscf",
            pw_cutoff=pw_cutoff,
            kppa=kppa[iter],
            pseudo=pseudo,
        )

        # reuse the nscf settings from the W convergence
        filename_nscf = qe_runner.qe_pw_run(
            calc_data_nscf,
            qe_write.write_pw,
            ncpu,
            kwargs={"n_bands": bnd_start, "input_dft": "lda"},
        )

        # create a Yambo subdirectory
        if not os.path.exists(f"kppa{kppa[iter]:d}"):
            os.mkdir(f"kppa{kppa[iter]:d}")

        # p2y step with the output redirected to the Yambo directory
        os.chdir(f"out/{id}.save")
        os.system(f"p2y -O ../../kppa{kppa[iter]:d}/")

        # move to the subdirectory for the Yambo calculation
        os.chdir(f"../../kppa{kppa[iter]:d}")

        # Yambo setup
        yambo_helper.generate_yambo_input_setup()
        os.system("yambo")

        # get the total number of k-points
        num_kpt = yambo_helper.get_num_kpt("r_setup")

        # read the r_setup to find where the direct gap is situated
        kpt_bnd_idx = yambo_helper.get_gamma_gap_parameters("r_setup")
        if kpt_bnd_idx[2] < num_elec / 2:
            print("Metallic states are present...", flush=True)
            open("metallic_states.txt", "w").close()
            kpt_bnd_idx[2] = int(num_elec / 2)
            kpt_bnd_idx[3] = int(num_elec / 2 + 1)

        # create a new subdirectory
        if not os.path.isdir("g0w0"):
            os.mkdir("g0w0")
        os.chdir("g0w0")

        # create the input file and start the calculation
        f_name = yambo_write.write_g0w0(
            bnd_start,
            cut_start,
            bnd_start,
            kpt_bnd_idx,
        )
        os.system(f"mpirun -np {ncpu} yambo -F {f_name}.in -J {f_name} -I ../")
        kgrid_gap.append(yambo_helper.get_minimal_gw_gap(f_name, kpt_bnd_idx))
        z = yambo_helper.get_z_factor(f_name)

        # go back up to the main directory
        os.chdir("../../")

        # append the output file
        kpt_step_time = int(np.ceil(time.time() - start_time_kpt))
        kstr = f"{k_grid[iter][0]:d}x{k_grid[iter][1]:d}x{k_grid[iter][2]:d}"
        with open(f"{id}_gw_conv.txt", "a+") as f:
            f.write(f"{kppa[iter]:<4d} \t {kstr:<9}  {gw.num_kpt:<5d}  ")
            f.write(
                f"{bnd_start:<5d}  {bnd_start:<5d}  {cut_start:<6d}  "
                + f"{kgrid_gap[iter]:<2.5f}  {1:<3d}  "
            )
            f.write(f"{kpt_step_time:<8d}  {ncpu:<4d}\n")

        # append the results to the list
        gw_list.append(
            [
                {"kppa": kppa[iter]}, # kppa
                {"kpt_grid": kstr}, # kpt grid
                {"num_kpts": gw.num_kpt}, # number of irreducible k-points
                {"X_bnds": bnd_start}, # number of bands in X
                {"G_bnds": bnd_start}, # number of bands in G
                {"X_cutoff": cut_start}, # cutoff for X (Ry)
                {"gamma_gap": kgrid_gap[iter]}, # gap at the Gamma point in eV
                {"num_gw_calc": iter + 1}, # number of GW calculations
                {"wall_time": kpt_step_time}, # wall time in seconds
                {"num_cores": ncpu}, # number of cores
                {"z-factor": z}, # z-factor
            ]
        )

        # calculate the difference in the band gap
        diff_gap = np.abs(kgrid_gap[iter] - kgrid_gap[iter - 1])

        # log message
        print(
            f"k-point grid: {kstr:>8s} "
            + f"(kppa = {kppa[iter]:>5d}) "
            + f"Gap = {kgrid_gap[-1]:6f} eV (Delta = {diff_gap:.6f} eV)",
            flush=True,
        )

        # convergence condition
        if diff_gap < conv_thr:
            print("\nGW Convergence achieved.", flush=True)
            break

        # break condition
        if iter + 1 > max_iter:
            f.write("CONVERGENCE FAILED. TOO MANY K-POINT GRID STEPS.")
            open("gw_kpt_conv_failed.txt", "w").close()
            raise Exception("Convergence failed.")

        # next k-point grid iteration
        iter += 1

    # log message for the total wall time
    end_time = time.time()
    conv_time = end_time - start_time
    print(f"Wall time: {conv_time:7.2f} s", flush=True)

    gw_dict.update(
        {
            "gw_kppa": kppa[-1],
            "gw_kpt_grid": k_grid[-1],
            "conv_list": gw_list,
            "conv_time": conv_time,
        }
    )

    # update the database entry
    cse.parameters.update({"gw_conv_lda": gw_dict})

    # add the GW band range for future GW calculations on the whole band structure
    gw_band_range = qe_helper.qe_get_gw_band_range(num_elec, f"out/{id:s}.xml")
    cse.parameters.update({"gw_band_range_lda": gw_band_range})

    # cleanup the directory
    shutil.rmtree("out")
    for k in kppa[1:-1]:
        if os.path.isfile(f"kppa{k}_cs/l_setup"):
            os.remove(f"kppa{k}/l_setup")
        os.remove(f"kppa{k}/r_setup")
        shutil.rmtree(f"kppa{k}/SAVE")
        shutil.rmtree(f"kppa{k}/g0w0/LOG")
        shutil.rmtree(f"kppa{k}/g0w0/{f_name:s}")
    if os.path.isfile(f"kppa{kppa[-1]}_cs/l_setup"):
        os.remove(f"kppa{kppa[-1]}/l_setup")
    os.remove(f"kppa{kppa[-1]}/r_setup")
    shutil.rmtree(f"kppa{kppa[-1]}/g0w0/LOG")
    shutil.rmtree(f"kppa{kppa[-1]}/g0w0/{f_name:s}")
    shutil.rmtree(f"kppa{kppa[-1]}/SAVE")

    # go back to main matarials directory
    os.chdir("../")

    return cse

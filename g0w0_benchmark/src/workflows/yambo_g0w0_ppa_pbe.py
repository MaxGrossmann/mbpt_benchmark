"""
This workflow performs a final GW calculation using the parameters obtained from 
'yambo_g0w0_conv_lda.py'. To achieve the most accurate benchmark results, this 
workflow increases all parameters by one "step".
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
import src.utils.yambo_helper as yambo_helper
import src.utils.yambo_write as yambo_write
from src.utils.calc_data_class import calc_data
from src.utils.basic_utils import get_kpt_grid
from src.utils.yambo_helper import get_direct_gw_gap


def yambo_g0w0_ppa_pbe(cse, base_dir, ncores, calc_dir, db_entry_path):

    # cpu cores
    ncpu = ncores

    # material id out of cse
    id = cse.parameters["id"]

    # print the workflow name
    print(f"\nStarting: {os.path.basename(__file__)}", flush=True)

    # workflow directory
    wf_dir = os.path.join(calc_dir, id, "yambo_g0w0_pbe")
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

    # get structure and ibrav from JSON
    structure = cse.structure
    ibrav = cse.parameters["ibrav"]
    num_elec = cse.parameters["num_elec_pbe"]

    """
    G0W0 calculation with converged parameters at gamma point
    """
    
    # start timing
    start_time = time.time()

    # get converged parameters
    bnd_step = cse.parameters["gw_conv_pbe"]["gw_bnd_step"]
    cut_step = cse.parameters["gw_conv_pbe"]["gw_cut_step"]
    kppa_step = 10
    bnd_final = int(cse.parameters["gw_conv_pbe"]["w_conv_grid"][-1][0] + bnd_step)
    cut_final = int(cse.parameters["gw_conv_pbe"]["w_conv_grid"][-1][1] + cut_step)
    kppa_scf = cse.parameters["qe_conv_pbe"]["qe_conv_steps"][-1][1]
    kppa = cse.parameters["gw_conv_pbe"]["conv_list"][-1][0]["kppa"]
    pw_cutoff = cse.parameters["gw_conv_pbe"]["pwcutoff"]

    # increase the k-point density until the k-point grid changes once
    k_grid = [np.array(get_kpt_grid(structure, kppa))]
    k_grid.append(np.array(get_kpt_grid(structure, kppa + kppa_step)))
    iter = 1
    while np.sum(np.abs(k_grid[iter] - k_grid[iter - 1])) == 0:
        kppa += kppa_step
        k_grid.append(np.array(get_kpt_grid(structure, kppa)))
        iter += 1

    """
    Quantum ESPRESSO
    """

    # initial scf calculation
    print(
        f"\nRunning a SCF with: cutoff = {pw_cutoff} Ry, k-point-density = {kppa_scf}",
        flush=True,
    )
    calc_data_scf = calc_data(
        structure,
        id=id,
        ibrav=ibrav,
        calc_type="scf",
        pw_cutoff=pw_cutoff,
        kppa=kppa_scf,
        pseudo=os.path.join(base_dir, "pseudo/PBE"),
    )
    filename_scf = qe_runner.qe_pw_run(
        calc_data_scf,
        qe_write.write_pw,
        ncpu,
    )

    # create nscf calculation based on scf calculation and run it
    print(
        f"\nRunning a SCF and NSCF with: cutoff = {pw_cutoff} Ry, k-point-density = {kppa}",
        flush=True,
    )
    calc_data_nscf = calc_data(
        structure,
        id=id,
        ibrav=ibrav,
        calc_type="nscf",
        pw_cutoff=pw_cutoff,
        kppa=kppa,
        pseudo=os.path.join(base_dir, "pseudo/PBE"),
    )
    filename_nscf = qe_runner.qe_pw_run(
        calc_data_nscf,
        qe_write.write_pw,
        ncpu,
        kwargs={"n_bands": bnd_final},
    )

    """
    Yambo
    """

    path = "g0w0"

    # create a Yambo subdirectory
    if not os.path.exists(path):
        os.mkdir(path)

    # p2y step with the output redirected to the Yambo directory
    os.chdir(f"out/{id}.save")
    os.system(f"p2y -O ../../{path}/")

    # move to the subdirectory for the Yambo calculation
    os.chdir(f"../../{path}")

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

    print(
        f"\nRunning G0W0 calculation with: N_bnd = {bnd_final}, G_cut = {cut_final} Ry, k-point-density = {kppa}",
        flush=True,
    )

    # create the input file and start the calculation
    f_name = yambo_write.write_g0w0(
        int(bnd_final),
        int(cut_final),
        int(bnd_final),
        kpt_bnd_idx,
    )
    os.system(f"mpirun -np {ncpu} yambo -F {f_name}.in -J {f_name} -I ../")
    kgrid_gap = yambo_helper.get_minimal_gw_gap(f_name, kpt_bnd_idx)

    # get GW direct gap
    gap = get_direct_gw_gap(f_name)

    # get z factor
    z = yambo_helper.get_z_factor(f_name)

    # get scissor
    scissor = yambo_helper.get_scissor(f_name=f_name)

    # go back up to the main directory
    os.chdir("../../")

    # create the output files
    with open(f"{id}_gw_final.txt", "w+") as f:
        f.write(
            f"kppa \t kpt \t    nk_ir  bnd_x  bnd_g  cutsrc  gap      #GW  time (s)  ncpu\n"
        )

    # append the output file
    kpt_step_time = int(np.ceil(time.time() - start_time))
    kstr = f"{k_grid[-1][0]:d}x{k_grid[-1][1]:d}x{k_grid[-1][2]:d}"
    with open(f"{id}_gw_final.txt", "a+") as f:
        f.write(f"{kppa:<4d} \t {kstr:<9}  {num_kpt:<5d}  ")
        f.write(
            f"{bnd_final:<5d}  {bnd_final:<5d}  {cut_final:<6d}  "
            + f"{kgrid_gap:<2.5f}  {1:<3d}  "
        )
        f.write(f"{kpt_step_time:<8d}  {ncpu:<4d}\n")

    # log message for the total wall time
    end_time = time.time()
    calc_time = end_time - start_time
    print(f"Wall time: {calc_time:7.2f} s", flush=True)

    gw_dict = {
        "gw_direct_gap": gap,
        "kppa": kppa,
        "kpt_grid": kstr,
        "num_kpt": num_kpt,
        "bands_in X": bnd_final,
        "bands_in_G": bnd_final,
        "cutoff": cut_final,
        "z_factor": z,
        "scissor": scissor,
        "ncpu": ncpu,
        "calc_time": calc_time,
    }

    # update the database entry
    cse.parameters.update({"g0w0_ppa_pbe": gw_dict})

    # add the GW band range for future GW calculations on the whole band structure
    gw_band_range = qe_helper.qe_get_gw_band_range(num_elec, f"out/{id:s}.xml")
    cse.parameters.update({"gw_band_range_pbe": gw_band_range})

    # print finale band gap
    print(
        f"G0W0 calculation at Gamma point finished with a direct band gap at gamma point: {gap}",
        flush=True,
    )

    # cleanup the directory
    shutil.rmtree("out")
    shutil.rmtree("g0w0/SAVE")
    shutil.rmtree(f"g0w0/g0w0/{f_name:s}")

    # go back to main matarials directory
    os.chdir("../")

    return cse

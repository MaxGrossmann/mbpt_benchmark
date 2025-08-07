"""
Functions that can start and restart Yambo workflows.
Primarily used for the GW convergence calculations.
"""

# external imports
import os
import shutil
import numpy as np

# local imports
import src.utils.qe_write as qe_write
import src.utils.qe_runner as qe_runner
import src.utils.yambo_helper as yambo_helper
from src.utils.basic_utils import uc_vol_au
from src.utils.calc_data_class import calc_data
from src.utils.yambo_gw_conv_class import conv_data

def yambo_run_gw_conv_cs(
    calc_data_scf,
    n_bands,
    kppa,
    gw,
    base_dir,
    ncpu,
    bnd_start=200,
    bnd_step=100,
    cut_start=4,
    cut_step=4,
    conv_thr=0.01,
    input_dft="pbe",
):
    """
    Starts a nscf in QE, creates a directory for a GW calculation and starts the coordinate search (CS) GW convergence algorithm.
    This function is useful to restart a convergence workflow if the number
    of bands needed to converge needs to be increased above the number used in the nscf step.
    INPUT:
        calc_data:          Class that contains the data for a QE calculation
        nbands:             Starting value for the maximum number of bands, this will be increased if too small
        kppa:               k-point density for the nscf calculation (0 = gamma point)
        gw:                 Convergence class used for restarting, else set to None
        base_dir:           Needed to find get the correct path to the pseudopotential files
        ncpu:               Number of cpu cores used for a mpi calls
        bnd_start:          Starting number of bands
        cut_start:          Starting cutoff
        bnd_step:           Steps in the number of bands
        cut_step:           Steps in the cutoff
        conv_thr:           Convergence threshold for the direct gap
        input_dft:          Pseudopotential setting for Quantum Espresso (default: "pbe")
    OUTPUT:
        conv_flag:          Flag, True if the convergence was successful
        bnd_increase_flag:  Flag that indicates if the number of bands should be increased
        gw:                 Convergence class used for restarting, i.e. when the maximum number of bands needs to be increased
        filename_scf:       Filename of the scf calculation
        filename_nscf:      Filename of the nscf calculation
        pw_cutoff:          Cutoff used for the last nscf calculation
    """

    # adjust the number of bands to fit with the number of cores
    if n_bands < bnd_start:
        print(
            "\nIncreasing n_bands to be compatible with the starting point...",
            flush=True,
        )
        n_bands = bnd_start + 3 * bnd_step
    n_bands = n_bands + n_bands % ncpu

    # unit cell volume to estimate the cutoff for the number of wanted bands
    vol = uc_vol_au(calc_data_scf.structure)

    # estimated the cutoff for the number of wanted bands
    arb_factor = 1.5 # seems to work well
    pw_cutoff = np.max(
        [
            int(np.ceil(((arb_factor * 8 * np.pi**2 * n_bands) / vol) ** (2 / 3))),
            calc_data_scf.pw_cutoff,
        ]
    )

    # redo the scf so the cutoff of the scf and nscf are the same
    # important because otherwise charge density is not correct
    calc_data_scf.pw_cutoff = pw_cutoff
    filename_scf = qe_runner.qe_pw_run(
        calc_data_scf,
        qe_write.write_pw,
        ncpu,
        kwargs={"input_dft": f"{input_dft:s}"},
    )

    # create nscf calculation based on QE and band gap workflow
    calc_data_nscf = calc_data(
        calc_data_scf.structure,
        id=calc_data_scf.id,
        ibrav=calc_data_scf.ibrav,
        calc_type="nscf",
        pw_cutoff=pw_cutoff,
        kppa=kppa,
        pseudo=os.path.join(base_dir, "pseudo"),
    )

    # try the nscf with a lot of bands, if it fails reduce the number of bands until it converges
    while True:
        print(f"\nCurrent nscf bands: {n_bands}", flush=True)
        filename_nscf = qe_runner.qe_pw_run(
            calc_data_nscf,
            qe_write.write_pw,
            ncpu,
            errorHandling=False,
            kwargs={"n_bands": n_bands, "input_dft": f"{input_dft:s}"},
        )

        # check if the nscf crashed (happens when there are too many bands)
        with open(f"{filename_nscf}.out", "r") as f:
            nscf_out_str = f.read()
        if "JOB DONE." in nscf_out_str or (n_bands - 4 * ncpu <= 64): # safety net
            break
        else:
            n_bands = n_bands - 4 * ncpu
            filename_nscf = qe_runner.qe_pw_run(
                calc_data_nscf,
                qe_write.write_pw,
                ncpu,
                errorHandling=False,
                kwargs={"n_bands": n_bands, "input_dft": f"{input_dft:s}"},
            )

    # create a Yambo subdirectory
    if not os.path.exists(f"kppa{kppa:d}_cs"):
        os.mkdir(f"kppa{kppa:d}_cs")

    # p2y step with the output redirected to the Yambo directory
    os.chdir(os.path.join(f"out/{calc_data_scf.id}.save"))
    os.system(f"p2y -O ../../kppa{kppa:d}_cs/")

    # move to the subdirectory for the Yambo calculation
    os.chdir(f"../../kppa{kppa:d}_cs")

    # Yambo setup
    yambo_helper.generate_yambo_input_setup()
    os.system("yambo")
    path_to_rsetup = os.getcwd()

    # cs algorithm
    if not os.path.isdir("g0w0_cs"):
        os.mkdir("g0w0_cs")
    os.chdir("g0w0_cs")

    if gw is None:
        gw = conv_data(
            ncpu,
            path_to_rsetup,
            conv_thr=conv_thr,
            bnd_start=bnd_start,
            bnd_step=bnd_step,
            cut_start=cut_start,
            cut_step=cut_step,
            cut_max=46, # hard coded ... larger is computationally to expense
        )
    if gw is not None:
        gw.bnd_max = n_bands
    conv_flag, bnd_increase_flag = gw.run_convergence()

    # clean up the directory
    gw.convergence_cleanup()

    # back to the starting directory
    os.chdir("../../")

    # clear up the Yambo directory
    if os.path.isfile(f"kppa{kppa:d}_cs/l_setup"):
        os.remove(f"kppa{kppa:d}_cs/l_setup")
        os.remove(f"kppa{kppa:d}_cs/r_setup")
    shutil.rmtree(f"kppa{kppa:d}_cs/SAVE")

    # if the convergence failed delete the scf file, nscf file and output directory from QE
    if not conv_flag:
        os.remove(filename_scf + ".in")
        os.remove(filename_scf + ".out")
        os.remove(filename_nscf + ".in")
        os.remove(filename_nscf + ".out")
        for file in os.listdir(os.path.join("out", f"{id}.save")):
            if "wfc" in file:
                os.remove(os.path.join("out", f"{id}.save", file))

    return conv_flag, bnd_increase_flag, gw, filename_nscf, pw_cutoff

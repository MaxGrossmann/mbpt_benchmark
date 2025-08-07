"""
Class for our GW convergence algorithm.
See: https://doi.org/10.1038/s41524-024-01311-9   
"""

# external imports
import os
import time
import shutil
import numpy as np

# local import
import src.utils.yambo_helper as yambo_helper
import src.utils.yambo_write as yambo_write

class conv_data:
    """
    Class which contains data about the GW convergence in Yambo
    ncores:             Number of cores for each calculation
    path_to_rsetup:     Relative path to the Yambo setup file
    conv_thr:           Convergence threshold for the direct gap
    bnd_start:          Starting point for the number of bands
    cut_start:          Starting point for the cutoff
    bnd_step:           Steps in the number of bands
    cut_step:           Steps in the cutoff
    cut_max:            Maximum cutoff
    """

    def __init__(
        self,
        ncores,
        path_to_rsetup,
        conv_thr=0.01,
        bnd_start=200,
        cut_start=6,
        bnd_step=50,
        cut_step=2,
        cut_max=46,
    ):
        """
        Function that initializes all important parameters for the CS convergence algorithm.
        """
        # defaults
        self.ncores = ncores
        self.path_to_rsetup = path_to_rsetup
        self.bnd_start = bnd_start
        self.cut_start = cut_start
        self.conv_thr = conv_thr
        self.bnd_step = bnd_step
        self.cut_step = cut_step
        self.cut_max = cut_max
        self.z_factor = []

        # complete the path to r_setup
        self.rsetup = os.path.join(self.path_to_rsetup, "r_setup")

        # get the number of electrons in the unit cell
        self.num_elec = yambo_helper.get_num_electrons(self.rsetup)
        if self.num_elec & 0x1:
            raise Exception("Uneven number of electrons in the unit cell!")

        # get the maximum number of bands
        self.bnd_max = yambo_helper.get_max_bands(self.rsetup)

        # get the total number of q-points
        self.num_kpt = yambo_helper.get_num_kpt(self.rsetup)

        # read the r_setup to find where the direct gap is situated
        self.kpt_bnd_idx = yambo_helper.get_gamma_gap_parameters(self.rsetup)
        if self.kpt_bnd_idx[2] < self.num_elec / 2:
            print("Metallic states are present...", flush=True)
            open("metallic_states.txt", "w").close()
            self.kpt_bnd_idx[2] = int(self.num_elec / 2)
            self.kpt_bnd_idx[3] = int(self.num_elec / 2 + 1)

        # dummy for the input file names
        self.fn = []

        # initialize grid variable to keep the convergence progress for plotting
        self.grid = []

        # initialize to time the indiviual steps on the grid
        # (good for timing predictions for real high-throughput calculations)
        self.grid_time = []

    def run_convergence(self):
        """
        Main function that runs the CS convergence algorithm.
        All results are stored inside the class.
        """
        # handmade coordinate search type algorithm
        print("\nCoordinate search convergence:", flush=True)

        # reference calculation at the starting point (only done when starting fresh)
        if not self.grid:
            print("Reference calculation at starting point:", flush=True)
            if self.bnd_start > self.bnd_max:
                print(
                    "Not enough bands for the starting point calculation...", flush=True
                )
                return False, True  # conv_flag, bnd_increase_flag
            f_name = yambo_write.write_g0w0(
                self.bnd_start, self.cut_start, self.bnd_start, self.kpt_bnd_idx
            )
            self.fn.append(f_name)
            step_time = time.time()
            os.system(
                f"mpirun -np {self.ncores} yambo -F {f_name}.in -J {f_name} -I ../"
            )
            self.grid_time.append(time.time() - step_time)
            self.gap_current = yambo_helper.get_direct_gw_gap(f_name)

            # z factor
            z = yambo_helper.get_z_factor(f_name)
            self.z_factor.append(z)

            print(
                f"{self.bnd_start:<4d} bands, {self.cut_start:<2d} Ry, Gap = {self.gap_current:6f} eV",
                flush=True,
            )
            self.grid.append([self.bnd_start, self.cut_start, self.gap_current])

        # parameters when doing the first start
        if len(self.grid_time) == 1:
            self.bnd = self.bnd_start
            self.cut = self.cut_start
            self.gap_diag = self.gap_current
            self.iter = 1
        diff_gap = 1 # some random value larger than the convergence threshold
        while True:
            if (self.bnd + self.bnd_step > self.bnd_max) and (
                self.cut + self.cut_step > self.cut_max
            ):
                print("Maximum calculation parameters were exceeded...", flush=True)
                return False, False # conv_flag, bnd_increase_flag
            print(f"\nIteration {self.iter}", flush=True)
            while True:
                if self.bnd + self.bnd_step > self.bnd_max:
                    print("Maximum number of bands was exceeded...", flush=True)
                    return False, True # conv_flag, bnd_increase_flag
                self.bnd += self.bnd_step
                f_name = yambo_write.write_g0w0(
                    self.bnd, self.cut, self.bnd, self.kpt_bnd_idx
                )
                self.fn.append(f_name)
                step_time = time.time()
                os.system(
                    f"mpirun -np {self.ncores} yambo -F {f_name}.in -J {f_name} -I ../"
                )
                self.grid_time.append(time.time() - step_time)
                new_gap = yambo_helper.get_direct_gw_gap(f_name)
                self.grid.append([self.bnd, self.cut, new_gap])
                delta_gap = np.abs(self.gap_current - new_gap)
                z = yambo_helper.get_z_factor(f_name)
                self.z_factor.append(z)

                print(
                    f"{self.bnd:<4d} bands, {self.cut:<2d} Ry, Gap = {new_gap:6f} eV (Delta = {delta_gap:6f} eV)",
                    flush=True,
                )
                self.gap_current = new_gap
                if delta_gap < self.conv_thr:
                    break
            while True:
                if self.cut + self.cut_step > self.cut_max:
                    print("Maximum cutoff was exceeded...", flush=True)
                    return False, False # conv_flag, bnd_increase_flag
                self.cut += self.cut_step
                f_name = yambo_write.write_g0w0(
                    self.bnd, self.cut, self.bnd, self.kpt_bnd_idx
                )
                self.fn.append(f_name)
                step_time = time.time()
                os.system(
                    f"mpirun -np {self.ncores} yambo -F {f_name}.in -J {f_name} -I ../"
                )
                self.grid_time.append(time.time() - step_time)
                new_gap = yambo_helper.get_direct_gw_gap(f_name)
                self.grid.append([self.bnd, self.cut, new_gap])
                delta_gap = np.abs(self.gap_current - new_gap)
                self.grid_time.append(time.time() - step_time)
                new_gap = yambo_helper.get_direct_gw_gap(f_name)
                self.grid.append([self.bnd, self.cut, new_gap])
                delta_gap = np.abs(self.gap_current - new_gap)
                z = yambo_helper.get_z_factor(f_name)
                self.z_factor.append(z)

                print(
                    f"{self.bnd:<4d} bands, {self.cut:<2d} Ry, Gap = {new_gap:6f} eV (Delta = {delta_gap:6f} eV)",
                    flush=True,
                )
                self.gap_current = new_gap
                if delta_gap < self.conv_thr:
                    break
            print("\nDiagonal gap comparison:")
            print(f"Previous   = {self.gap_diag:6f} eV", flush=True)
            print(f"Current    = {self.gap_current:6f} eV", flush=True)
            diff_gap = np.abs(self.gap_diag - self.gap_current)
            print(f"Difference = {diff_gap:.6f} eV", flush=True)
            self.diag_diff = diff_gap
            if diff_gap <= self.conv_thr:
                break
            self.gap_diag = self.gap_current
            self.iter += 1

        # save the final convergence point
        self.final_point = np.array([self.bnd, self.cut, self.gap_current])

        return True, False # conv_flag, bnd_increase_flag

    def convergence_cleanup(self):
        files = os.listdir()
        for f in files:
            if os.path.isdir(f):
                shutil.rmtree(f)

"""
Here we store little functions that are
a) Only necessary for Yambo and
b) Don't start calculations on their own
"""

# external imports
import re
import os
import numpy as np
import pathlib as pl

def get_gamma_gap_parameters(path_to_rsetup):
    """
    Reads the r_setup file in the Yambo directory from a given path path_to_rsetup.
    Returns the k-point and band indices of the gap at the gamma point.
    INPUT:
        path_to_rsetup:     Path to the r_setup file
    OUTPUT:
        kpt_bnd_idx:        kpt_idx1, kpt_idx2, bnd_idx1, bnd_idx2
    """

    # read the input file
    with open(path_to_rsetup, "r") as f:
        setup_str = f.read()

    # array for gap location in the band structure
    kpt_bnd_idx = np.zeros(4, dtype=int)

    # get the vbm and cbm index of the direct band gap at the gamma point
    kpt_bnd_idx[2] = int(
        re.findall(r"\d+", re.findall(r"Filled Bands[ \t]+:[ \t]+\d+", setup_str)[0])[0]
    )
    kpt_bnd_idx[3] = kpt_bnd_idx[2] + 1

    # get the k-point index of the direct band gap at the gamma point
    kpt_bnd_idx[0] = 1
    kpt_bnd_idx[1] = 1

    return kpt_bnd_idx

def get_direct_gap_parameters(path_to_rsetup):
    """
    Reads the r_setup file in the Yambo directory from a given path path_to_rsetup.
    Returns the k-point and band indices of the direct gap and the gap value.
    INPUT:
        path_to_rsetup:     Path to the r_setup file
    OUTPUT:
        direct_gap:         Value of the direct gap in eV
        kpt_bnd_idx:        kpt_idx1, kpt_idx2, bnd_idx1, bnd_idx2
    """

    # read the input file
    with open(path_to_rsetup, "r") as f:
        setup_str = f.read()

    # array for gap location in the band structure
    kpt_bnd_idx = np.zeros(4, dtype=int)

    # get the vbm and cbm index of the direct band gap
    kpt_bnd_idx[2] = int(
        re.findall(r"\d+", re.findall(r"Filled Bands[ \t]+:[ \t]+\d+", setup_str)[0])[0]
    )
    kpt_bnd_idx[3] = kpt_bnd_idx[2] + 1

    # get the k-point index of the direct band gap
    kpt_bnd_idx[0] = int(
        re.findall(
            r"\d+",
            re.findall(r"Direct Gap localized at k[ \t]+:[ \t]+\d+", setup_str)[0],
        )[0]
    )
    kpt_bnd_idx[1] = kpt_bnd_idx[0]

    # get the direct band gap
    direct_gap = float(
        re.findall(
            r"\d+.\d+", re.findall(r"Direct Gap[ \t]+:[ \t]+\d+.\d+", setup_str)[0]
        )[0]
    )

    return direct_gap, kpt_bnd_idx

def get_indirect_gap_parameters(path_to_rsetup):
    """
    Reads the r_setup file in the Yambo directory from a given path path_to_rsetup.
    Returns the k-point and band indices of the indirect gap and the gap value.
    INPUT:
        path_to_rsetup:     Path to the r_setup file
    OUTPUT:
        indirect_gap:       Value of the indirect gap in eV
        kpt_bnd_idx:        kpt_idx1, kpt_idx2, bnd_idx1, bnd_idx2
    """

    # read the input file
    with open(path_to_rsetup, "r") as f:
        setup_str = f.read()

    # array for gap location in the band structure
    kpt_bnd_idx = np.zeros(4, dtype=int)

    # get the vbm and cbm index of the indirect band gap
    kpt_bnd_idx[2] = int(
        re.findall(r"\d+", re.findall(r"Filled Bands[ \t]+:[ \t]+\d+", setup_str)[0])[0]
    )
    kpt_bnd_idx[3] = kpt_bnd_idx[2] + 1

    # get the k-point index of the indirect band gap
    matches = re.findall(
        r"\d+",
        re.findall(r"Indirect Gap between kpts[ \t]+:[ \t]+\d+[ \t]+\d+", setup_str)[0],
    )
    kpt_idx = [int(m) for m in matches]
    kpt_bnd_idx[0] = kpt_idx[0]
    kpt_bnd_idx[1] = kpt_idx[1]

    # get the indirect band gap
    indirect_gap = float(
        re.findall(
            r"\d+.\d+", re.findall(r"Indirect Gap[ \t]+:[ \t]+\d+.\d+", setup_str)[0]
        )[0]
    )

    return indirect_gap, kpt_bnd_idx

def get_num_electrons(path_to_rsetup):
    """
    Reads the r_setup file in the Yambo directory from a given path path_to_rsetup.
    Returns the number of electrons in the system.
    INPUT:
        path_to_rsetup:     Path to the r_setup file
    OUTPUT:
        num_elec:           Number of electrons in the system
    """

    # read the input file
    with open(path_to_rsetup, "r") as f:
        setup_str = f.read()

    # get the number of electrons in the system
    num_elec = int(
        re.findall(r"\d+", re.findall(r"Electrons[ \t]+:[ \t]+\d+", setup_str)[0])[0]
    )

    return num_elec

def get_max_bands(path_to_rsetup):
    """
    Reads the r_setup file in the Yambo directory from a given path path_to_rsetup.
    Returns the maximum number of bands available.
    INPUT:
        path_to_rsetup:     Path to the r_setup file
    OUTPUT:
        max_bands:          Maximum number of bands available
    """

    # read the input file
    with open(path_to_rsetup, "r") as f:
        setup_str = f.read()

    # get the maximum number of bands available
    max_bands = int(
        re.findall(r"\d+", re.findall(r"Bands[ \t]+:[ \t]+\d+", setup_str)[0])[0]
    )

    return max_bands

def get_num_kpt(path_to_rsetup):
    """
    Reads the r_setup file in the Yambo directory from a given path path_to_rsetup.
    Returns the maximum number of bands available.
    INPUT:
        path_to_rsetup:     Path to the r_setup file
    OUTPUT:
        num_kpt:            Total number of k-points
    """

    # read the input file
    with open(path_to_rsetup, "r") as f:
        setup_str = f.read()

    # get number of k-points
    num_kpt = int(
        re.findall(r"\d+", re.findall(r"IBZ K-points :[ \t]+\d+", setup_str)[0])[0]
    )

    return num_kpt

def get_direct_gw_gap(f_name, dft_flag=False):
    """
    Reads the GW .qp output file from a GW convergence calculation at the direct gap.
    This functions always operates in the currect directory.
    Returns the new value of the direct gap after the GW calculation.
    INPUT:
        f_name:             File name of the .qp output file
        dft_flag:           Flag to also output the DFT gap
    OUTPUT:
        direct_gap:         Direct gap energy (eV)
        direct_dft_gap:     Direct gap energy from DFT (eV) (optional)
    """

    # parse the output file
    data = np.loadtxt(f"o-{f_name}.qp", comments="#")

    # calculate the direct gap
    direct_gap = (data[1, 2] + data[1, 3]) - (data[0, 2] + data[0, 3])

    if dft_flag:
        direct_dft_gap = data[1, 2] - data[0, 2]
        return direct_gap, direct_dft_gap
    else:
        return direct_gap

def get_minimal_gw_gap(f_name, kpt_bnd_idx):
    """
    Reads the GW .qp output file from a GW convergence calculation on the full q-grid.
    kpt_bnd_idx:    contains the k-point and band indicies where the minimal gap is located
    This functions always operates in the currect directory.
    Returns the new value of the minimal (indirect/direct) gap after the GW calculation.
    INPUT:
        f_name:             File name of the .qp output file
        kpt_bnd_idx:        Contains the k-point and band indicies where the minimal gap is located
    OUTPUT:
        min_gap:            Minimal band gap energy (eV)
    """

    # parse the output file
    data = np.loadtxt(f"o-{f_name}.qp", comments="#")

    # calculate the minimal gap
    k1 = data[data[:, 0] == kpt_bnd_idx[0]]
    vbm = np.sum(k1[k1[:, 1] == kpt_bnd_idx[2], :][0][2:4])
    k2 = data[data[:, 0] == kpt_bnd_idx[1]]
    cbm = np.sum(k2[k2[:, 1] == kpt_bnd_idx[3], :][0][2:4])
    min_gap = cbm - vbm

    return min_gap

def get_z_factor(f_name):
    """
    Reads the results file "r-..." from GW calculations. Find all bands with associated Z-factor.

    INPUT:
        f_name:     File name of results file
    OUTPUT:
        z:          Z-factor as array (first column: band; second column: Re(Z), third column: Im(Z))
    """

    files = pl.Path(os.getcwd()).glob(f"r-{f_name:s}*")

    iter = 0

    for file in files:
        iter += 1
        if iter > 2:
            raise Exception("Multiple similar files ...")
        with open(file, "r") as f:
            res_f = f.read()
            z = np.array(
                re.findall(
                    r"B=(\d+).+Re\(Z\)=(.?\d+\.\d+E?[+-]?\d+?).+?Im\(Z\)=(.?\d+\.\d+E?[+-]?\d+?)",
                    res_f,
                )
            )
            z = z.astype(float)

    return z

def get_scissor(f_name):
    """
    Get the scissor of a G0W0 calculation at Gamma-Point.

    INPUT:
        f_name:         File name from Yambo .qp file
    OUTPUT:
        sc:             Scissor operator value (eV)
    """

    data = np.loadtxt(f"o-{f_name}.qp", comments="#")

    sc = data[1, 3] - data[0, 3]

    return sc

def generate_yambo_input_setup():
    """
    Generates a Yambo setup file. Add to this setup file calculation methode NoDiagSC.
    """

    with open("yambo.in", "w") as f:
        f.write("setup\nNoDiagSC")

"""
Contains all functions that do not perform calculations.
"""

# external imports
import re
import os
import sys
import json
import shutil
import spglib
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from monty.json import MontyDecoder
from importlib.resources import files
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.core import Element, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

# internal imports
from qsgw_workflow.utils.system_config import execute_command, execute_command_timeout, get_execution_mode

# load the plot style
style_path = files("qsgw_workflow.files").joinpath("plotstyle.mplstyle")
plt.style.use(style_path)

class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def ry2ev(x):
    """
    Convert eV to Ry.
    """
    return 13.6057039763 * x

def combine_n_lists(sym_path_list):
    """
    Converts a list with broken up symmetry paths, e.g., [["X", "K", "W"], ["G", "L", "U"]]
    to a combined list usable for the x-tick labels of a band structure plot.
    Example: [["X", "K", "W"], ["G", "L", "U"]] -> ["X", "K", "W,G", "L", "U"]
    INPUT:
        sym_path_list:      List with broken up symmetry paths, e.g., [["X", "K", "W"], ["G", "L", "U"]]
    OUTPUT:
        combined :          List for the x-tick labels of a band structure plot
    """
    if len(sym_path_list) == 1:
        return sym_path_list[0]
    combined = sym_path_list[0][:-1]
    for i in range(len(sym_path_list) - 1):
        combined.append(sym_path_list[i][-1] + "," + sym_path_list[i + 1][0])
        combined += sym_path_list[i + 1][1:-1]
    combined.append(sym_path_list[-1][-1])
    return combined

# mapping of greek characters to their latex equivalents (you may need to extend this...)
greek_letters = {
    "Γ": r"\Gamma",
    "Σ": r"\Sigma",
}

# mapping of unicode subscript digits to their normal digit characters
subscript_map = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3",
    "₄": "4", "₅": "5", "₆": "6", "₇": "7",
    "₈": "8", "₉": "9"
}

def convert_subscripts(subs):
    """
    Convert a sequence of Unicode subscript digits to a normal digit string.
    """
    return "".join(subscript_map.get(ch, ch) for ch in subs)

def replace_greek_with_subscript(match):
    """
    Replace a Greek letter followed by subscript digits with LaTeX format.
    """
    letter = match.group(1)
    subs = match.group(2)
    latex_letter = greek_letters.get(letter, letter)
    subs_normal = convert_subscripts(subs)
    return fr"${latex_letter}_{{{subs_normal}}}$"

def replace_non_greek_with_subscript(match):
    """
    Replace a non-Greek letter followed by subscript digits with LaTeX format.
    """
    letter = match.group(1)
    subs = match.group(2)
    subs_normal = convert_subscripts(subs)
    return f"{letter}$_{{{subs_normal}}}$"

def replace_single_greek(match):
    """
    Replace a single Greek letter with its LaTeX equivalent.
    """
    letter = match.group(0)
    return f"${greek_letters.get(letter, letter)}$"

def replace_greek_and_subscripts(text):
    """
    Function to convert the band structure path obtained from the Questaal
    output into a format needed for nice plots.
    """
    # pattern for greek letter(s) followed by unicode subscript digits
    greek_chars = "".join(re.escape(ch) for ch in greek_letters.keys())
    pattern_greek_sub = re.compile(rf"([{greek_chars}])([₀-₉]+)")
    text = pattern_greek_sub.sub(replace_greek_with_subscript, text)
    # pattern for non-greek letter followed by unicode subscript digits
    pattern_nongreek_sub = re.compile(r"([A-Za-z])([₀-₉]+)")
    text = pattern_nongreek_sub.sub(replace_non_greek_with_subscript, text)
    # pattern for standalone greek letters
    pattern_single_greek = re.compile(rf"([{greek_chars}])")
    text = pattern_single_greek.sub(replace_single_greek, text)
    return text

def gen_init_str(id, structure):
    """
    Generates the Questaal 'init.mat' file string.
    INPUT:
        id:             Some identifier of the structure
        structure:      pymatgen structure object
    """
    # variable initialization
    lines = []
    # header, i.e., id and composition
    spg = structure.get_space_group_info()
    header = f"HEADER {id:s} {structure.composition.reduced_formula:s} {spg[0]:s}"
    lines.append(header)
    # lattice parameters that depend on the space group number
    lines.append("LATTICE")
    lines.append(f"#       SPCGRP={spg[1]:d}")
    a = structure.lattice.a
    b = structure.lattice.b
    c = structure.lattice.c
    alpha = structure.lattice.alpha
    beta = structure.lattice.beta
    gamma = structure.lattice.gamma
    lines.append(f"#       A={a:.5f}  B={b:.5f}  C={c:.5f}   ALPHA={alpha:.0f}  BETA={beta:.0f}  GAMMA={gamma:.0f}")
    lines.append(f"% const a={a:.5f}")
    lines.append("        ALAT={a}  UNITS=A")
    plat = structure.lattice.matrix / a
    for i in range(3):
        prefix = "        PLAT=    " if i == 0 else "                 "
        line = prefix + "    ".join(f"{x:.8f}" for x in plat[i])
        lines.append(line)
    # site, specie and fractional coordinates
    lines.append("SITE")
    for site in structure:
        species = site.specie.symbol
        x, y, z = site.frac_coords
        lines.append(f"     ATOM={species:<2}       X=    {x:.8f}    {y:.8f}    {z:.8f}")
    return "\n".join(lines)

def standardize_cell(structure):
    """
    Get the standardized primitive structure from the input structure.
    INPUT:
        structure:              Structure that is supposed to be processed
    OUTPUT:
        standard_structure:     Standardized primitive structure
    """
    # atomic positions have to be specified by scaled positions for spglib.
    lattice = structure.lattice.matrix
    scaled_positions = structure.frac_coords
    numbers = [i.specie.Z for i in structure.sites]
    cell = (lattice, scaled_positions, numbers)
    lattice, scaled_positions, numbers = spglib.standardize_cell(
        cell, to_primitive=True, symprec=1e-5, # same tolerance that 'blm' uses internally
    )
    s = Structure(lattice, numbers, scaled_positions)
    standard_structure = s.get_sorted_structure()
    return standard_structure

def cse2init(struct_path, struct_name):
    """
    Load a structure from a ComputedStructureEntry and convert the contained structure into a Questaal init file.
    The resulting 'init.mat' file is created in the current working directory.
    INPUT:
        struct_path:        Absolute path to the structure file
        struct_name:        Name of the material
    OUTPUT:
        struct:             pymatgen structure object
    """
    struct_path = f"{struct_path:s}/{struct_name:s}.json"
    cse = load_db_entry(struct_path)
    with open(struct_path, "r") as f:
        json_dict = json.load(f)
    cse = ComputedStructureEntry.from_dict(json_dict)
    struct = cse.structure
    init_str = gen_init_str(struct_name, struct)
    with open(f"init.mat", "w") as f:
        f.write(init_str)
    return struct

def get_kpt_grid(struct, kppa):
    """
    Generates the k-point-grid for a structure with given k-point density
    INPUT:
        struct:         pymatgen struct object
        kppa:           k-point density. If equal to 0, return a gamma-only grid
    OUTPUT:
        kgrid:          A 3x1 list which specifies the number of subdivision in reciprocal space in each dimension
                        (i.e., the normal way to format k-grids). The k-grid is rounded up to the nearest even number
                        to ensure a good parallelization
    """
    kpts = Kpoints.automatic_density(structure=struct, kppa=kppa, force_gamma=True)
    if kppa == 0: # for gamma-only calculations
        kgrid = [1, 1, 1]
    else:
        kgrid = [i if i % 2 == 0 else i + 1 for i in kpts.kpts[0]]
    return kgrid

def increase_kppa(struct, kppa, delta_kppa=10):
    """
    Increase the kppa until a more dense k-grid is found.
    INPUT:
        struct:         pymatgen structure object
        kppa:           k-point density
        delta_kppa:     k-point density step size (keep this small, but the exact value is not so important)
    OUTPUT:
        kppa:           Increased k-point density
    """
    kpts = get_kpt_grid(struct, kppa)
    kppa += delta_kppa
    new_kpts = get_kpt_grid(struct, kppa)
    # increase the kppa until a more dense k-grid is found
    # the difference of 4 ensures that we do not go from, e.g., a (4x4x4) k-grid to a (6x4x4) k-grid
    while np.abs(np.sum(np.array(kpts) - np.array(new_kpts))) < 4:
        kppa += delta_kppa
        new_kpts = get_kpt_grid(struct, kppa)
    kpts = new_kpts
    return kppa

def fix_basis(struct):
    """
    This function reads and updates the 'ctrl.mat' file.
    It removes/updates the 'pz= ...' segment (and adjusts lmx, lmxa)
    for problematic elements. For elements like Hf and Tl, the entire
    basis set info is on one line, while for Ta and W it spans two lines.
    Changes are mostly taken from https://www.questaal.org/images/LMTO_settings.txt
    and from https://molmod.ugent.be/sites/default/files/deltadftcodes/supplmat/SupplMat-QUESTAAL.pdf
    Hf is "really special"... in some compounds the basis set breaks when the 5f orbital is included...
    (This function could be written in a more comprehensive way in the future...)
    INPUT:
        struct:     pymatgen structure object
    """
    special_hafnium_mats = [
        "HfSe2"
    ]
    if struct.composition.reduced_formula in special_hafnium_mats:
        hafnium_pz = "pz=0\t15.9312\t0\t0"
    else:
        hafnium_pz = "pz=0\t15.9312\t0\t5.3"
    with open("ctrl.mat", "r") as f:
        lines = f.readlines()
    modified_lines = []
    i = 0
    max_lmxa = 0
    while i < len(lines):
        line = lines[i]
        # Tl = one-line entry
        # the automatic basis for Tl seems broken, so we have to fix it
        if "atom=Tl" in line:
            line = re.sub(r"pz=.*", "pz=0\t0\t15.9148", line)
            line = re.sub(r"lmx=\d+", "lmx=4", line)
            line = re.sub(r"lmxa=\d+", "lmxa=5", line)
            modified_lines.append(line)
            i += 1
        # for Hf, Ta, and W, we include the 4f and 5f states
        # Hf has 4f as its valence, so we include 5f as a high-lying local orbital (HLLO)
        # Ta and W have 5f as valence, so we include 4f as a core orbital
        # for all three, we exclude the high-lying d orbitals because they interfere with the low-lying f orbitals
        # Hf = one-line entry
        elif "atom=Hf" in line:
            line = re.sub(r"pz=.*", hafnium_pz, line)
            line = re.sub(r"lmx=\d+", "lmx=4", line)
            line = re.sub(r"lmxa=\d+", "lmxa=6", line)
            modified_lines.append(line)
            i += 1
        # Ta = two-line entry
        elif "atom=Ta" in line:
            new_pz = "pz=0\t15.9346\t0\t14.9365"
            line = re.sub(r"lmx=\d+", "lmx=4", line)
            line = re.sub(r"lmxa=\d+", "lmxa=5", line)
            # check if 'pz=' is on the same line as the atom keyword
            if "pz=" in line:
                line = re.sub(r"pz=.*", new_pz, line)
                modified_lines.append(line)
                i += 1
            else:
                # the current line does not contain 'pz=', so it should be on the next line
                modified_lines.append(line)
                if i + 1 < len(lines):
                    next_line = lines[i+1]
                    if "pz=" in next_line:
                        next_line = re.sub(r"pz=.*", new_pz, next_line)
                    modified_lines.append(next_line)
                    i += 2
                else:
                    i += 1
        # W = two-line entry
        elif "atom=W" in line:
            new_pz = "pz=0\t15.9376\t0\t14.9423"
            line = re.sub(r"lmx=\d+", "lmx=4", line)
            line = re.sub(r"lmxa=\d+", "lmxa=5", line)
            # check if 'pz=' is on the same line as the atom keyword
            if "pz=" in line:
                line = re.sub(r"pz=.*", new_pz, line)
                modified_lines.append(line)
                i += 1
            else:
                # the current line does not contain 'pz=', so it should be on the next line
                modified_lines.append(line)
                if i + 1 < len(lines):
                    next_line = lines[i+1]
                    if "pz=" in next_line:
                        next_line = re.sub(r"pz=.*", new_pz, next_line)
                    modified_lines.append(next_line)
                    i += 2
                else:
                    i += 1
        else:
            modified_lines.append(line)
            i += 1
        # gather all lmxa values
        if "atom=" in line:
            lmxa = int(re.search(r"lmxa=(\d+)", line).group(1))
            max_lmxa = max(max_lmxa, lmxa)
    # lmxa must be the same for all sites when performing a QSGW calculation
    for i, line in enumerate(modified_lines):
        if "atom=" in line:
            modified_lines[i] = re.sub(r"lmxa=\d+", f"lmxa={max_lmxa:d}", line)
    with open("ctrl.mat", "w") as f:
        f.writelines(modified_lines)

def fix_qloc(struct):
    """
    Adjust the criteria to decide which orbitals should be included in the valence as local orbitals.
    If the fraction of free atomic wavefunction charge outside the augmentation radius exceeds 'qloc',
    the orbital is included as a local orbital. See: https://www.questaal.org/about/verification/.
    During the benchmark, we identified a number of materials for which this breaks the basis set.
    For this reason they are excluded. This is just a quick and dirty fix...
    INPUT:
        struct:     pymatgen structure object
    """
    # quick and dirty fix for the benchmark...
    broken_basis_mats = [
        "AlCuO2",
        "GaCuO2",
        "LaCuO2",
        "ScCuO2",
        "Cu2O",
    ]
    if struct.composition.reduced_formula in broken_basis_mats:
        return
    with open("ctrl.mat", "r") as f:
        ctrl_str = f.read()
    ctrl_str = re.sub("eloc=-2.5", "eloc=-2.5 qloc=0.002", ctrl_str)
    with open("ctrl.mat", "w") as f:
        f.write(ctrl_str)

def check_basis():
    """
    Setup, parse and check the basis set parameters.
    (This just calls 'lmfa' and 'lmchk'...)
    References:
    https://www.questaal.org/tutorial/lmf/basis_set/#definition-of-the-basis-set (2a)
    https://www.questaal.org/tutorial/lmf/basis_set/#definition-of-the-basis-set (4)
    OUTPUT:
        gmax:           Cutoff energy from lmfa (local orbitals)
    """
    execute_command(f"lmfa ctrl.mat --usebasp > lmfa.log")
    gmax = os.popen("grep 'GMAX=' lmfa.log").read()
    # take the largest gmax (valence vs. local orbitals) for more accurate QSGW results
    gmax = np.max([float(m) for m in re.findall(r"\d+.\d+", gmax)])
    # set the suggested plane-wave cutoff 'gmax' in the ctrl.mat file
    set_gmax(gmax)
    # check the basis set
    lmchk()
    return gmax

def lmchk():
    """
    Run lmchk and check the packing fraction and maximum overlap.
    (Assumes that 'blm' and 'lmfa' have been run before.)
    """
    execute_command(f"lmchk ctrl.mat > lmchk.log")
    with open("lmchk.log", "r") as f:
        s = f.read()
    pf = re.findall(r"(Sum of sphere volumes\= \d+.\d+ \(\d+.\d+\))", s)[0]
    pf = float(re.findall(r"\d+.\d+", pf)[1])
    print(f"    packing fraction = {pf:.2f}", flush=True)
    if pf < 0.1: # random estimate... in general larger packing fractions are better
        open("small_sphere_packing_fraction.txt", "w").close
        sys.exit(f"The packing fraction should be larger than 0.1!")
    ovlp = re.findall(r"max ovlp \= .*", s)[0]
    ovlp = np.array([float(m) for m in re.findall(r"[-+]?\d*\.?\d+", ovlp)])
    ovlp_str = "    maximum overlap  = "
    ovlp_str += " ".join([f"{m:.2f}%" for m in ovlp])
    print(ovlp_str, flush=True)
    if any(ovlp > 1.0): # for GW calculations, 1% is acceptable, for LDA, 10% is acceptable
        open("large_sphere_overlap.txt", "w").close
        sys.exit(f"The overlap should be smaller than 1%!")

def set_gmax(gmax):
    """
    Adjust the plane-wave cutoff 'gmax' in the 'ctrl.mat' file.
    INPUT:
        fpts:           Number of frequency points
        window:         Frequency window (Ha)
    """
    print(f"    Setting gmax = {gmax:.1f}", flush=True)
    with open("ctrl.mat", "r") as f:
        ctrl_str = f.read()
    ctrl_str = re.sub(r"gmax=[^\s]+", f"gmax={gmax:.1f}", ctrl_str)
    with open("ctrl.mat", "w") as f:
        f.write(ctrl_str)

def init_basis(struct):
    """
    Initializes the basis for a Questaal calculation.
    Runs the following executables: 'blm', 'lmfa', 'lmchk'.
    INPUT:
        struct:         pymatgen structure object
    OUTPUT:
        gmax:           Cutoff energy from 'lmfa' (including local orbitals)
    """
    print("\nBasis setup:", flush=True)
    # setup for the ctrl file (blm)
    kpts = get_kpt_grid(struct, 1000) # arbitrary k-grid
    kpt_str = f"--nk~{kpts[0]:d},{kpts[1]:d},{kpts[2]:d}" # placeholder in the input file
    """
    Comments about the AUTOBAS settings:
    - eloc:   See https://www.questaal.org/about/verification/, the default is -2.0 Ry
              but for Ca and Sc (where the atomic states lie at -2.06 and -2.47Ryd) -2.5 Ry is better
              (same applies to Ru, Rh, Ir and Pt, whose the 4p and 5p states extend significantly into the interstitial)
    """
    # see https://www.questaal.org/about/verification/
    bas_str = "--autobas~eloc=-2.5"
    """
    Comments about the GW settings:
    - emax:   See https://doi.org/10.1103/PhysRevB.108.165104 Tab. XII, good default 
    - nvbse:  Always prepare the 'ctrl.mat' file for a BSE calculation
    - ncbse:  See above
    - rsmin:  See https://www.questaal.org/docs/input/commandline/, internal algorithm returns
              'gcutb' and 'gcutx' for the worst case, so we can decrease the cost of the calculation a bit
              (For now, I decided not to do this and instead go for maximum accuracy, but maybe later...)
    """
    gw_str = "--gw~emax=3.0~nvbse=1~ncbse=1"
    """
    Comments about the general settings (see 'blm' call):
    - tidy:          Yymmetrizes structure to obey the symmetry of the space group (useful for POSCARs)
    - optics:        Prepares the input for the calculation of the dielectric function 
    - findes~floats: Find and add empty spheres to the basis set to be used as floating orbitals (good for QSGW calculations)
    """
    # run 'blm' and add empty sphere if possible (improves QSGW calculations)
    # ('--mix~nit=50' sets the maximum number of DFT iterations to 50)
    execute_command(
        f"blm init.mat {kpt_str:s} {bas_str:s} {gw_str:s} --mix~nit=50 --findes~floats --optics --ctrl=ctrl > blm.log"
    )
    with open("blm.log", "r") as f:
        blm_log = f.read()
    if "Exit -1  CVPLAT: could not calculate platcv." in blm_log:
        sys.exit("The symmetry finder got confused!")
    # check the number of empty spheres added by 'blm'
    with warnings.catch_warnings(): # catch new numpy warning associated with 'np.loadtxt'...
        warnings.simplefilter("ignore")
        site_str = np.loadtxt(f"site.mat", comments=["#", "%"], dtype=str)
    es_counter = 0
    for row in site_str:
        if re.match(r"^E\d*$", row[0]):
            es_counter += 1
    print(f"    Added {es_counter:d} empty spheres (as floating orbitals)", flush=True)
    # fix the auto-generated base for some elements that cause problems
    fix_basis(struct)
    # adjust the 'qloc', the command line option '--autobas~qloc=-0.002' does not work for us
    # (it is set to 0.002 based on https://www.questaal.org/about/verification/)
    fix_qloc(struct)
    # check the basis set parameters ('lmfa' + 'lmchk') and get an good estimate for the plane-wave cutoff 'gmax'
    gmax = check_basis()
    return gmax

def set_dft_kgrid(dft_kpts, indent=True):
    """
    Adjust the DFT k-grid in the 'ctrl.mat' file.
    INPUT:
        dft_kpts:       DFT k-grid as a list
        indent:         Flag to indent the command line message
    """
    if indent:
        print(
            f"\n    Setting the DFT k-grid to ({dft_kpts[0]:d}x{dft_kpts[1]:d}x{dft_kpts[2]:d}).",
            flush=True,
        )
    else:
        print(
            f"\nSetting the DFT k-grid to ({dft_kpts[0]:d}x{dft_kpts[1]:d}x{dft_kpts[2]:d}).",
            flush=True,
        )
    with open("ctrl.mat", "r") as f:
        ctrl_str = f.read()
    ctrl_str = re.sub(r"nk1=[^\s]+", f"nk1={dft_kpts[0]:d}", ctrl_str)
    ctrl_str = re.sub(r"nk2=[^\s]+", f"nk2={dft_kpts[1]:d}", ctrl_str)
    ctrl_str = re.sub(r"nk3=[^\s]+", f"nk3={dft_kpts[2]:d}", ctrl_str)
    with open("ctrl.mat", "w") as f:
        f.write(ctrl_str)

def set_gw_kgrid(gw_kpts, indent=True):
    """
    Adjust the GW k-grid in the 'ctrl.mat' file.
    INPUT:
        gw_kpts:        GW k-grid as a list
        indent:         Flag to indent the command line message
    """
    if indent:
        print(
            f"    Setting the self energy (QSGW) k-grid to ({gw_kpts[0]:d}x{gw_kpts[1]:d}x{gw_kpts[2]:d}).",
            flush=True,
        )
    else:
        print(
            f"Setting the self energy (QSGW) k-grid to ({gw_kpts[0]:d}x{gw_kpts[1]:d}x{gw_kpts[2]:d}).",
            flush=True,
        )
    with open("ctrl.mat", "r") as f:
        ctrl_str = f.read()
    ctrl_str = re.sub(r"nkgw1=[^\s]+", f"nkgw1={gw_kpts[0]:d}", ctrl_str)
    ctrl_str = re.sub(r"nkgw2=[^\s]+", f"nkgw2={gw_kpts[1]:d}", ctrl_str)
    ctrl_str = re.sub(r"nkgw3=[^\s]+", f"nkgw3={gw_kpts[2]:d}", ctrl_str)
    with open("ctrl.mat", "w") as f:
        f.write(ctrl_str)

def set_eps_window(fpts=5001, emax=5):
    """
    Adjust the energy window for the calculation of the dielectric function.
    INPUT:
        fpts:           Number of frequency points
        emax:           Maximum energy (Ha)
    """
    with open(f"ctrl.mat", "r") as f:
        s = f.read()
    s = re.sub("npts=1001", f"npts={fpts:d}", s)
    s = re.sub("window=0,1", f"window=0,{emax:.2f}", s)
    with open(f"ctrl.mat", "w") as f:
        f.write(s)

def grep_ehf(fname):
    """
    Parse the last Harris-Foulkes energy from an lmf output file.
    INPUT:
        fname:          Name of the output file
    OUTPUT:
        ehf:            Harris-Foulkes energy (Ry)
    """

    match = os.popen(f"grep ehk= {fname:s} | tail -1").read()
    ehf_match = re.findall(r"ehf=[+-]?\d+.\d+", match)[0]
    ehf = float(ehf_match.split("=")[1])
    return ehf

def get_vbm_idx():
    """
    Get the index of the highest valence band through the number of electrons used for the DFT.
    (Reads the 'dft.log' file.)
    OUTPUT:
        vbm_idx:        Zero-based band index of the highest valence band
    """
    with open("dft.log", "r") as f:
        log_str = f.read()
    match = re.findall(r"\d+.\d+ electrons", log_str)[0] # just to the first entry
    num_elec = float(re.findall(r"\d+.\d+", match)[0])
    vbm_idx = int(np.ceil(num_elec / 2)) - 1 # zero-based index
    print(
        f"\nThe system contains {num_elec:.1f} electrons -> vbm_idx = {vbm_idx:d} (zero-based).",
        flush=True,
    )
    return vbm_idx

def get_gap(fname):
    """
    Parses the band gap from a log file generated by lmf.
    INPUT:
        fname:          Name of a lmf log file
    OUTPUT:
        gap:            band gap in eV or -1 if the material is a metal
    """
    out = os.popen(f'grep -A 3 " BZWTS : --- Tetrahedron Integration ---" "{fname:s}" | tail -3').read()
    m = re.search(
        r"gap\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)" + 
        r"(?:\s*Ry\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?",
        out,
    )
    if not m:
        return -1.0
    gap = float(m.group(2)) 
    return gap

def get_unique_atoms(struct):
    """
    Get a list of all unique elements and their atomic numbers, sorted by their atomic number.
    INPUT:
        struct:         pymatgen structure object
    OUTPUT:
        elem_list:      List of [elem, Z], sorted by Z
    """
    elements = np.unique([site.specie.name for site in struct.sites])
    atm_nums = []
    for elem in elements:
        atm_nums += [Element(elem).Z]
    sort_idx = np.argsort(atm_nums)
    elem_list = [[elements[i], atm_nums[i]] for i in sort_idx]
    return elem_list

def parse_bandstructure(fname):
    """
    Parses a the band structure file with and without color weights.
    The output can be used for plots later on.
    (This parser does not work for spin-polarized calculations.)
    INPUT:
        fname:      Name of the band structure file
    OUTPUT:
        bs:         Dictionary with all information about the band structure
    """
    # load the file and parse the header
    with open(fname, "r") as file:
        file_contents = file.readlines()
    header = file_contents[0].strip().split()
    n_bands = int(header[0])
    e_fermi = float(header[1])
    n_colors = int(header[2])
    path_string = header[3].split("=")[1]
    # analyse the band structure path and obtain the axis labels
    paths = [s.split("-") for s in path_string.split("|")]
    symbol_list = []
    path_labels = []
    n_points = 0
    for p in paths:
        symbols = []
        for s in p:
            symbol_list += [s]
            symbols += [s]
            n_points += 1
        path_labels += [symbols]
    # needed for the tick labels with broken band structure paths
    tick_labels = combine_n_lists(path_labels)
    # prepare the file contents
    file_contents = file_contents[1:]
    # useful variables
    bands_per_line = 10 # believe this is hardcoded in Questaal
    band_lines = int(np.ceil(n_bands / bands_per_line))
    # this part parses the band structure
    # the Questaal 'bnds.mat' file format is cursed...
    # do not try to understand or change this part of the code...
    bs_paths = []
    idx = 0
    path_start = 0
    if n_colors == 0:
        while idx < n_points:
            path_dict = {}
            path_dict["label"] = symbol_list[idx]
            values = file_contents[path_start].strip().split()
            nk = int(values[0])
            if nk == 0:
                break
            path_end = nk * (band_lines + 1) + 1 + path_start
            k_points = []
            bands = []
            temp_bands = []
            flag = 0
            for i in range(path_start + 1, path_end):
                values = [
                    float(num) for num in re.findall(r"-?\d+\.\d+", file_contents[i])
                ]
                if flag == 0:
                    k_points += [[float(v) for v in values]]
                    if len(temp_bands) == n_bands:
                        bands.append(temp_bands)
                    temp_bands = []
                    flag = 1
                elif flag == 1:
                    temp_bands.extend([float(v) for v in values])
                    if len(temp_bands) == n_bands:
                        flag = 0
            bands.append(temp_bands) # bands for the last k-point
            path_dict["nk"] = nk
            path_dict["k_points"] = np.array(k_points)
            path_dict["bands"] = ry2ev(np.array(bands) - e_fermi)
            path_dict["colors"] = []
            bs_paths.append(path_dict)
            path_start = path_end
            idx += 1
    else:
        while idx < n_points:
            path_dict = {}
            path_dict["label"] = symbol_list[idx]
            values = file_contents[path_start].strip().split()
            nk = int(values[0])
            if nk == 0:
                break
            path_end = (n_colors + 1) * nk * (band_lines + 1) + 1 + path_start
            k_points = []
            bands = []
            colors = []
            temp_bands = []
            temp_colors = []
            counter = 0
            flag = 0
            for i in range(path_start + 1, path_end):
                values = [
                    float(num) for num in re.findall(r"-?\d+\.\d+", file_contents[i])
                ]
                if flag == 0:
                    temp_k_point = [float(v) for v in values]
                    flag = 1
                    if len(temp_bands) == n_bands and counter == 0:
                        k_points += [temp_k_point]
                        bands.append(temp_bands)
                        counter += 1
                        flag = 2
                    if len(temp_colors) == n_colors * n_bands and counter == n_colors:
                        colors.append(
                            np.array(temp_colors).reshape([n_colors, n_bands])
                        )
                        counter = 0
                        temp_colors = []
                        flag = 1
                    if len(temp_colors) >= n_bands and counter > 0:
                        counter += 1
                        flag = 2
                    temp_bands = []
                elif flag == 1 and counter == 0:
                    temp_bands.extend([float(v) for v in values])
                    if len(temp_bands) == n_bands:
                        flag = 0
                elif flag == 2 and counter > 0:
                    temp_colors.extend([float(v) for v in values])
                    if len(temp_colors) == counter * n_bands:
                        flag = 0
            colors.append(np.array(temp_colors).reshape([n_colors, n_bands]))
            color_array = np.zeros([nk, n_bands, n_colors])
            for i in range(nk):
                color_array[i, :, :] = colors[i].T
            path_dict["nk"] = nk
            path_dict["k_points"] = np.array(k_points)
            path_dict["bands"] = ry2ev(np.array(bands) - e_fermi)
            path_dict["colors"] = color_array
            bs_paths.append(path_dict)
            path_start = path_end
            idx += 1
    # dictionary with all information about the band structure
    bs = {
        "n_colors": n_colors,
        "n_bands": n_bands,
        "tick_labels": tick_labels,
        "bs_paths": bs_paths,
    }
    return bs

def plot_bs(ax, bs, lcs="k-", deco=True, cflag=[0, 0, 0]):
    """
    Useful function when plotting band structures.
    INPUT:
        ax:             Figure axis
        bs:             Band structure object (output of the 'parse_bandstructure()' function)
        lcs:            Line color and style
        deco:           Flag to add vertical lines at the high-symmetry points and axis labels
        cflag:          List containing 0 or 1 indicating which color projections to enable
    """
    n_colors = bs["n_colors"]
    n_bands = bs["n_bands"]
    tick_labels = bs["tick_labels"]
    tick_labels = [replace_greek_and_subscripts(item) for item in tick_labels]
    bs_paths = bs["bs_paths"]
    x0 = 0
    label_idx = []
    cmax = 0.0
    for path in bs_paths:
        c = np.array(path["colors"])
        if len(c) == 0:
            cflag = [0, 0, 0]
            break
        if c.max() > cmax:
            cmax = c.max()
    for path in bs_paths:
        label_idx += [x0]
        nk = path["nk"]
        x = np.arange(x0, x0 + nk)
        if n_colors == 0 or sum(cflag) == 0:
            for i in range(n_bands - 1):
                y = np.array(path["bands"])[:, i]
                ax.plot(x, y, lcs)
        else:
            c = np.array(path["colors"])
            c[c < 0] = 0.0
            c /= c.max()
            for i in range(n_bands - 1):
                y = np.array(path["bands"])[:, i]
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                plot_colors = np.zeros([nk, 3])
                for j in range(c.shape[2]):
                    if cflag[j]:
                        plot_colors[:, j] = c[:, i, j]
                lc = LineCollection(
                    segments,
                    colors=plot_colors,
                    path_effects=[path_effects.Stroke(capstyle="round")],
                )
                ax.add_collection(lc)
        x0 = x[-1]
    label_idx += [x[-1]]
    if deco:
        ax.axhline(y=0, color="k", linestyle="-.", lw=0.5, zorder=-1)
        for i in label_idx:
            ax.axvline(x=i, color="k", linestyle="-", lw=0.5, zorder=-1)
        ax.set_xticks(label_idx)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim([0, x[-1]])
        ax.set_ylabel("Energy (eV)")

def parse_pdos(fname):
    """
    The function assumes that the PDOS was calculated for all sites
    in the structure with 'lcut=2', i.e., '--pdos~mode=1~lcut=2'.
    From there the total DOS is obtained. The d-orbital PDOS for 
    all sites are summed up. The output can be used for plots later on.
    (This parser does not work for spin-polarized calculations.)
    INPUT:
        fname:           Name of the dos file
    OUTPUT:
        pdos:            Dictionary with all information about the PDOS
    """
    with open(fname, "r") as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    emin = float(header[0])
    emax = float(header[1])
    npts = int(header[2])
    nchan = int(header[3])
    if nchan == 1:
        sys.exit("\nThe parser does not work with single-channel DOS files!")
    efermi = float(header[5])
    energy = ry2ev(np.linspace(emin, emax, npts) - efermi)
    pdos = []
    for l in lines[1:]:
        values = l.strip().split()
        pdos.extend([float(v) for v in values])
    pdos = np.array(pdos).reshape((nchan, npts))
    tdos = np.sum(pdos, axis=0).flatten()
    return {"energy": energy, "tdos": tdos, "pdos": pdos}

def parse_eps(fname):
    """
    This function parses the dielectric tensor from an IPA calculation.
    The real part is obtained from the imaginary part using the Kramers-Kronig transformation.
    ('kkt' is currently only available in the Questaal development version, so you may need to change this...)
    INPUT:
        fname:          Name of the dos file
    OUTPUT:
        eps_imag:       Dictionary with all information about the dielectric tensor
    """
    execute_command(f"kkt -outfile=kkt.mat -units:out=ev {fname:s}")
    data = np.loadtxt(f"kkt.mat", comments=["#", "%"])
    execute_command("rm kkt.mat")
    omega = data[:, 0]
    eps_imag = data[:, [2, 4, 6]]
    eps_real = data[:, [1, 3, 5]]
    eps = {
        "omega": omega,
        "eps_imag": eps_imag,
        "eps_real": eps_real,
    }
    return eps

def calc_sc(eps1, eps2, emax=30.0):
    """
    The function to calculate the similarity coefficient between two the dielectric tensor.
    INPUT:
        eps1:           Dictionary from "parse_eps()"
        eps2:           Dictionary from "parse_eps()"
        emax:           Maximum energy (eV) considered for the calculation of the SC
    OUTPUT:
        sc:             Similarity coefficient
    """
    idx = eps1["omega"] < emax
    sc = np.zeros([3])
    for i in range(3):
        sc[i] = 1 - np.trapz(
            np.abs(eps1["eps_imag"][idx, i] - eps2["eps_imag"][idx, i])
        ) / np.trapz(eps1["eps_imag"][:, i])
    return np.mean(sc)

def create_pqmap(nnodes, ncores, fill=0.8, bse_flag=False, max_time=300.0):
    """
    Improve parallelization of the 'lmsig' code using an automatically generated 'pqmap'.
    (See: https://www.questaal.org/docs/code/userguide/ and https://www.questaal.org/tutorial/gw/hpc/)
    INPUT:
        nnodes:         Number of nodes to be used for the calculation
        ncores:         Number of cores to be used for the calculation
        fill:           https://www.questaal.org/tutorial/gw/hpc/#33-construction-of-pqmap
        bse_flag:       Flag to create the pqmap for a BSE calculation
        max_time:       Maximum seconds the code can take to create a pqmap
    OUTPUT:
        mpi_str:        Parallelization instructions for 'ncores' on a single (vanilla) node
    """
    # number of cores per node
    ncores_per_node = ncores // nnodes
    # helps changing the parallelization setting for the Noctua 2 supercomputer in Paderborn
    def change_m_str(m):
        n = [int(m) for m in re.findall(r"\d+", m)]
        n = [n[0], n[2]]
        return f"srun --export=ALL,OMP_NUM_THREADS={n[0]:d},MKL_NUM_THREADS={n[0]:d} -n {n[1]:d}"
    # get the global execution mode
    execution_mode = get_execution_mode()
    # create the 'pqmap' and a batch script
    if bse_flag:
        print(f"    Generating a 'pqmap-bse' for {ncores_per_node:d} cores per node and {nnodes:d} nodes.", flush=True)
        execute_command("rm -f pqmap-bse-[0-9]* batch-[0-9]*")
    else:
        print(f"    Generating a 'pqmap' for {ncores_per_node:d} cores per node and {nnodes:d} nodes.", flush=True)
        execute_command("rm -f pqmap-[0-9]* batch-[0-9]*")
    execute_command("lmfgwd ctrl.mat --job=-1 > pqmap.log")
    while True:
        if fill < 0.5:
            # run a calculation without 'pqmap', this will probably be slow, but better than doing nothing
            if bse_flag:
                print("    Unable to generate 'pqmap-bse'.", flush=True)
                execute_command("rm -f pqmap-bse-[0-9]* batch-[0-9]*")
            else:
                print("    Unable to generate 'pqmap'.", flush=True)
                execute_command("rm -f pqmap-[0-9]* batch-[0-9]*")
            m1=f"env OMP_NUM_THREADS={ncores_per_node:d} MKL_NUM_THREADS={ncores_per_node:d} mpirun -n 1"
            mn=f"env OMP_NUM_THREADS={nnodes:d} MKL_NUM_THREADS={nnodes:d} mpirun -n {ncores_per_node:d}"
            ml=f"env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -n {ncores:d}" # when using multiple nodes, 'lmf' and 'lmfgwd' can be very slow when using multiple threads....
            md=f"env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -n {ncores:d}" # when using multiple nodes, 'lmf' and 'lmfgwd' can be very slow when using multiple threads....
            mc=f"env OMP_NUM_THREADS={nnodes:d} MKL_NUM_THREADS={nnodes:d} mpirun -n {ncores_per_node:d}"
            if execution_mode == "noctua":
                m1 = change_m_str(m1)
                mn = change_m_str(mn)
                ml = change_m_str(ml)
                md = change_m_str(md)
                mc = change_m_str(mc)
            mx = mn
            mb = mn
            mpi_str = f'--mpirun-1 "{m1:s}" --mpirun-n "{mn:s}" --cmd-lm "{ml:s}" --cmd-gd "{md:s}" --cmd-vc "{mc:s}" --cmd-xc " {mx:s}" --cmd-bs "{mb:s}"'
            return mpi_str
        if bse_flag:
            pq_str = (
                f"lmfgwd ctrl.mat --job=0 --lmqp~rdgwin " +
                f"--batch~bsw~np={ncores_per_node:d}~pqmap@fill={fill:.1f}~nodes={nnodes:d}~vanilla{ncores_per_node:d}"
            )
        else:
            pq_str = (
                f"lmfgwd ctrl.mat --job=0 --lmqp~rdgwin " +
                f"--batch~np={ncores_per_node:d}~pqmap@fill={fill:.1f}~nodes={nnodes:d}~vanilla{ncores_per_node:d}"
            )
        timeout_flag = execute_command_timeout(pq_str, "pqmap.log", max_time=max_time)
        with open("pqmap.log", "r") as f:
            pqmap_log_str = f.read() # helps to check if the 'pqmap' was really generated
        if (not timeout_flag) and ("Exit 0" in pqmap_log_str):
            break
        fill -= 0.1
    # log the packing fraction
    with open("pqmap.log", "r") as f:
        pq_str = f.read()
    match = re.search(r"packing fraction \d+(?:\.\d+)?", pq_str)
    print(f"    (pqmap -> {match.group()})", flush=True)
    # parse the parallelization options from the batch script
    with open(f"batch-{ncores // nnodes:d}", "r") as f:
        batch_str = f.read()
    pattern = r"m\w+\s*=\s*\"(env\s+OMP_NUM_THREADS=\d+\s+MKL_NUM_THREADS=\d+\s+mpirun\s+-n\s+\d+)\""
    matches = re.findall(pattern, batch_str)
    m1 = matches[0]
    mn = matches[1]
    ml = matches[2]
    md = matches[3]
    if bse_flag:
        mc = matches[5] # skipping 'mb'
    else:
        mc = matches[4]
    # when using multiple nodes, 'lmf' and 'lmfgwd' can be very slow when using multiple threads....
    ml=f"env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -n {ncores:d}"
    md=f"env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -n {ncores:d}"
    # adjustments the parallelization strings for the Noctua 2 supercomputer in Paderborn
    if execution_mode == "noctua":
        m1 = change_m_str(m1)
        mn = change_m_str(mn)
        ml = change_m_str(ml)
        md = change_m_str(md)
        mc = change_m_str(mc)
    mx = mn
    mb = mn
    # 'lmgw.sh' parallelization input
    mpi_str = f'--mpirun-1 "{m1:s}" --mpirun-n "{mn:s}" --cmd-lm "{ml:s}" --cmd-gd "{md:s}" --cmd-vc "{mc:s}" --cmd-xc " {mx:s}" --cmd-bs "{mb:s}"'
    return mpi_str

def get_rsrnge():
    """
    Read the description of 'check_and_fix_bloch_sum' below.
    This function reads the current value of the 'rsrnge' parameter from the 'llmf-sym' file.
    OUTPUT:
        rsrnge:         Maximum range of connecting vectors for the real-space self energy
                        (https://www.questaal.org/docs/input/inputfile/)
    """
    with open("llmf-sym", "r") as f:
        lines = f.readlines()
    rsrnge = False
    for line in lines:
        if line.startswith(" hft2rs: make"):
            # regular expression to extract the number
            match = re.search(r"range\s*=\s*([\d.]+)", line)
            rsrnge = float(
                match.group(1)
            )  # convert to float for both integer and floating-point numbers
    if rsrnge:
        return rsrnge
    else:
        raise Exception("Could not find rsrnge!")

def set_rsrnge(rsrnge):
    """
    Read the description of 'check_and_fix_bloch_sum' below.
    This function sets the 'rsrnge' parameter in the 'ctrl.mat' file. 
    It first checks whether the corresponding line is already present, 
    and if so, updates it. Otherwise, it adds the line.
    INPUT:
        rsrnge:         Maximum range of connecting vectors for the real-space self energy
                        (https://www.questaal.org/docs/input/inputfile/)
    """
    with open("ctrl.mat", "r") as f:
        lines = f.readlines()
    # check if the line is already present, if so, just modify the value
    # modify the line containing 'rsrnge'
    for i, line in enumerate(lines):
        if line.strip().startswith("rsrnge"):
            # replace the value after '='
            parts = line.split("=")
            if len(parts) > 1:
                parts[
                    1
                ] = f" {rsrnge:.3f} # helps with the inexact inverse bloch transformation error\n"
                lines[i] = "=".join(parts)
            with open("ctrl.mat", "w") as f:
                lines = f.writelines(lines)
            return

    # else, add the line
    modified_lines = []
    new_line = f"      rsrnge={rsrnge:.3f} # helps with the inexact inverse bloch transformation error\n"
    in_ham_block = False
    ham_end_inserted = False
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("ham"):
            in_ham_block = True
            modified_lines += [line]
            continue
        if in_ham_block and not ham_end_inserted:
            modified_lines += [new_line]
            ham_end_inserted = True
            in_ham_block = False
        modified_lines += [line]
    with open("ctrl.mat", "w") as f:
        lines = f.writelines(modified_lines)

def check_and_fix_bloch_sum():
    """
    Run 'lmgw.sh' before using this function, otherwise the log file 'llmf-sym' will not exist!
    This function checks whether the bloch sum was inexact, if so, it calls 'get_rsrnge' to get 
    the current value, increases it by calling 'set_rsrnge', and then returns whether the Bloch sum was exact.
    OUTPUT:
        bloch_sum_error_flag:       0 if no error occurred, 1 if an error was found
    """
    if not os.path.exists("llmf-sym"):
        sys.exit("\nRun 'lmgw.sh' before using 'check_and_fix_bloch_sum', otherwise the log file 'llmf-sym' will not exist!")
    with open("llmf-sym", "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith(" Oops!  Bloch sum"):
            print(
                "        Inverse Bloch sum is inexact! Increasing RSRNGE by 2 and rounding up.",
                flush=True,
            )
            rsrnge = get_rsrnge()
            print(f"        Old RSRNGE = {rsrnge:.3f}", flush=True)
            rsrnge += 2
            rsrnge = int(np.ceil(rsrnge))
            print(f"        New RSRNGE = {rsrnge:.3f}", flush=True)
            set_rsrnge(rsrnge)
            return 1
    return 0

def get_bse_vb_manifold(bs, vbm_idx, lowest_energy=10, manifold_tolerance=0.5):
    """
    Find how many valence bands you need to include in the BSE Hamiltonian
    This function examines the band structure and counts, starting from the VBM
    (which is set to zero by definition, see 'parse_bandstructure()'), the number of valence 
    bands whose maximum energy is within the specified range from the VBM.
    INPUT:
        bs:                     Band structure object (see 'parse_bandstructure()' above)
        vbm_idx:                Index of the VBM
        lowest_energy:          Energy in eV (determines the number of VBs in the BSE Hamiltonian)
        manifold_tolerance:     If there is a band manifold around the lowest_energy,
                                we want to include the whole manifold, i.e. we want to close the manifold,
                                so here the manifold tolerance is there two determine when a new band manifold starts
    OUTPUT:
        counter:                Number of valence bands to include in the BSE Hamiltonian
    """
    energies = []
    for path in bs["bs_paths"]:
        energies.append(path["bands"])
    energies = np.vstack(energies)
    curr_band = vbm_idx
    counter = 1
    manifold_tolerance = 0.5
    while curr_band > 0:
        min_curr = np.min(energies[:, curr_band])
        max_lower = np.max(energies[:, curr_band - 1])
        if max_lower < -lowest_energy:
            if min_curr - max_lower < manifold_tolerance:
                counter += 1
                curr_band -= 1
            else:
                break
        else:
            counter += 1
            curr_band -= 1
    return counter

def get_bse_cb_manifold(bs, vbm_idx, highest_energy=10):
    """
    Find out how many conduction bands you need to include in the BSE Hamiltonian.
    This function examines the band structure and counts, starting from the CBM, 
    the number of conduction bands whose minimum energy is within the specified range from the CBM.
    INPUT:
        bs:                 Band structure object (see 'parse_bandstructure()' above)
        vbm_idx:            Index of the VBM
        highest_energy:     Energy in eV (determines the number of CBs in the BSE Hamiltonian)
    OUTPUT:
        counter:            Number of conduction bands to include in the BSE Hamiltonian
    """
    energies = []
    for path in bs["bs_paths"]:
        energies.append(path["bands"])
    energies = np.vstack(energies)
    num_bands = energies.shape[1]
    curr_band = vbm_idx + 1 # CBM index
    cbm = np.min(energies[:, curr_band])
    counter = 1
    while curr_band < num_bands:
        min_curr = np.min(energies[:, curr_band + 1])
        if (min_curr - cbm) > highest_energy:
            break
        else:
            counter += 1
            curr_band += 1
    return counter

def set_bse_bands(nv, nc):
    """
    Adjust the number of bands included in the BSE Hamiltonian in the 'ctrl.mat' file.
    INPUT:
        nv:             Number of valence bands
        nc:             Number of conduction bands
    """
    print(
        f"\nSetting the transition space for the BSE Hamiltonian (nv={nv:d}, nc={nc:d}).",
        flush=True,
    )
    with open("ctrl.mat", "r") as f:
        ctrl_str = f.read()
    ctrl_str = re.sub(r"nvbse=\d+", f"nvbse={nv:d}", ctrl_str)
    ctrl_str = re.sub(r"ncbse=\d+", f"ncbse={nc:d}", ctrl_str)
    with open("ctrl.mat", "w") as f:
        f.write(ctrl_str)

def clean_qsgw(name, rst_flag=True, indent=True):
    """
    BE VERY CAREFUL IN WHICH DIRECTORY YOU RUN THIS FUNCTION!
    Clear the working directory of files that may conflict with a prior QSGW/QSGW^ run.
    (This can also be used to clean the working directory while keeping most files, i.e., 'rst_flag=False'.)
    Reference: https://www.questaal.org/docs/code/userguide/#command-line-arguments-to-lmgwsh
    INPUT:
        name:               Name of the material (sanity check)
        rst_flag:           Flag to clean the working directory without keeping and with keep restart files
                            True:   clean everything except the input and basis set files
                            False:  keep most files, but remove large h5 files
        indent:             Flag to indent the command line message (True, False, "")
    """
    files = os.listdir()
    # safety feature (I once accidentally deleted my entire git repository using this function...)
    if os.getcwd().split("/")[-1] != name:
        sys.exit(
            "clean_qsgw(): The working directory is not the calculation directory!"
        )
    if rst_flag: # removes more than necessary I think, but we want to be safe...
        if indent == True:
            print("    Cleaning up the directory for a fresh QSGW/QSGW^ calculation.", flush=True)
        elif indent == False:
            print("\nCleaning up the directory for a fresh QSGW/QSGW^ calculation.", flush=True)
        for f in files:
            if (
                f.startswith("slurm-")
                or f.startswith("basp")
                or f.startswith("atm")
                or f.startswith("ctrl")
                or f.startswith("init")
                or f.startswith("site")
                or f.startswith("bnds_")
                or f.startswith("bs_")
                or f.startswith("dos_")
                or f.startswith("opt_")
                or f.endswith(".err")
                or f.endswith(".log")
                or f == "run.sh"
            ):
                continue
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)
    else:  # https://www.questaal.org/docs/code/userguide/ (keep most files for restarts)
        if indent == True:
            print(
                "    Cleaning up the directory of temporary QSGW/QSGW^ files and directories.",
                flush=True,
            )
        elif indent == False:
            print(
                "\nCleaning up the directory of temporary QSGW/QSGW^ files and directories.",
                flush=True,
            )
        # https://www.questaal.org/tutorial/gw/gw_dielectric/
        execute_command("touch meta bz.h5; rm -rf [0-9]*run meta mixm.mat mixsigma; lmgwclear")

def load_db_entry(db_path):
    """
    Load a database entry from a JSON file.
    Can also be used to load any ComputedStructureEntry from a JSON file.
    INPUT:
        db_path:            Path to the database
    OUTPUT:
        db_entry:           Dictionary of the database entry
    """
    decoder = MontyDecoder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db_entry = decoder.process_decoded(json.load(open(db_path, "r")))
    return db_entry

def save_db_entry(name, db_path, struct, param_dict, data_dict, encoder=NumpyEncoder):
    """
    Save a database entry to a JSON file.
    INPUT:
        name:               Name of the material
        db_path:            Path of a database file
        struct:             pymatgen structure object
        param_dict:         Dictionary with all calculation parameters
        data_dict:          Dictionary with all calculation results
        encoder:            Encoder for different object, e.g., numpy array in our case
    """
    print(f"Updating the database entry.", flush=True)
    # create a ComputedStructureEntry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cse = ComputedStructureEntry(
            struct,
            energy=0.0, # dummy value
            parameters=param_dict,
            data=data_dict,
            entry_id=name,
        )
        json_dict = cse.as_dict()
    # save the ComputedStructureEntry in a json file
    with open(db_path, "w") as f:
        json.dump(json_dict, f, cls=encoder)
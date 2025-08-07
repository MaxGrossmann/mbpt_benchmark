"""
Here we store little functions that are
a) Only necessary for QE and
b) Don't start calculations on their own (i.e. don't need qe_write)
"""

# external imports
import os
import re
import math
import spglib
import numpy as np
import xml.etree.ElementTree as ET
from pymatgen.core import Element
from pymatgen.io.pwscf import PWInput
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

# local imports
from src.utils.unit_conversion import ang2au, ev2ha, au2ang, ha2ev

class qe_PWInput(PWInput):
    """
    Initializes a PWSCF input file.
    Adapted version of the original pymatgen version by Miguel Marques (thanks again!)
    Read the original pymatgen code for details.
    (https://github.com/materialsproject/pymatgen/blob/v2025.5.28/src/pymatgen/io/pwscf.py#L22-L564)
    """

    def __str__(self):
        out = []
        site_descriptions = {}

        if self.pseudo is not None:
            site_descriptions = self.pseudo
        else:
            c = 1
            for site in self.structure:
                name = None
                for k, v in site_descriptions.items():
                    if site.properties == v:
                        name = k

                if name is None:
                    name = site.specie.symbol + str(c)
                    site_descriptions[name] = site.properties
                    c += 1

        def to_str(v):
            if isinstance(v, str):
                return f"'{v}'"
            if isinstance(v, float):
                return f"{str(v).replace('e', 'd')}"
            if isinstance(v, bool):
                if v:
                    return ".TRUE."
                return ".FALSE."
            return v

        for k1 in ["control", "system", "electrons", "ions", "cell"]:
            v1 = self.sections[k1]
            out.append(f"&{k1.upper()}")
            sub = []
            for k2 in sorted(v1.keys()):
                if isinstance(v1[k2], list):
                    n = 1
                    for l in v1[k2][: len(site_descriptions)]:
                        sub.append(f"  {k2}({n}) = {to_str(v1[k2][n - 1])}")
                        n += 1
                else:
                    sub.append(f"  {k2} = {to_str(v1[k2])}")
            if k1 == "system":
                if "ibrav" not in self.sections[k1]:
                    sub.append("  ibrav = 0")
                if "nat" not in self.sections[k1]:
                    sub.append(f"  nat = {len(self.structure)}")
                if "ntyp" not in self.sections[k1]:
                    sub.append(f"  ntyp = {len(site_descriptions)}")
            sub.append("/")
            out.append(",\n".join(sub))

        out.append("ATOMIC_SPECIES")
        for k, v in sorted(site_descriptions.items(), key=lambda i: i[0]):
            e = re.match(r"[A-Z][a-z]?", k).group(0)
            if self.pseudo is not None:
                p = v
            else:
                p = v["pseudo"]
            out.append(f"  {k}  {Element(e).atomic_mass:.4f} {p}")

        out.append("ATOMIC_POSITIONS crystal")
        if self.pseudo is not None:
            for site in self.structure:
                out.append(f"  {site.specie} {site.a:.8f} {site.b:.8f} {site.c:.8f}")
        else:
            for site in self.structure:
                name = None
                for k, v in sorted(site_descriptions.items(), key=lambda i: i[0]):
                    if v == site.properties:
                        name = k
                out.append(f"  {name} {site.a:.8f} {site.b:.8f} {site.c:.8f}")

        out.append(f"K_POINTS {self.kpoints_mode}")
        if self.kpoints_mode == "automatic":
            kpt_str = [f"{i}" for i in self.kpoints_grid]
            kpt_str.extend([f"{i}" for i in self.kpoints_shift])
            out.append(f"  {' '.join(kpt_str)}")
        elif (
            self.kpoints_mode == "crystal"
        ): # just used for wannier calculation in our case
            out.append(self.kpoints_grid)
        elif (
            self.kpoints_mode == "tpiba"
        ): # just used for effective mass calculation in our case
            out.append(self.kpoints_grid)
        elif self.kpoints_mode == "gamma":
            pass

        # Difference to original
        if "ibrav" not in self.sections["system"]:
            out.append("CELL_PARAMETERS angstrom")
            for vec in self.structure.lattice.matrix:
                out.append(f"  {vec[0]:.15f} {vec[1]:.15f} {vec[2]:.15f}")

        return "\n".join(out) + "\n\n"

def qe_standardize_cell(structure, symprec=1e-5):
    """
    Obtain the standard primitive structure from the input structure.
    INPUT:
        structure:      Structure that is supposed to be processed
        symprec:        Precision for symmetry prediction. Lower the value to get more precision
                        at a higher risk of not identifying symmetries
    OUTPUT:
        standard_structure: Standardized primitive structure
    """
    # Atomic positions have to be specified by scaled positions for spglib.
    lattice = structure.lattice.matrix
    scaled_positions = structure.frac_coords
    numbers = [i.specie.Z for i in structure.sites]
    cell = (lattice, scaled_positions, numbers)
    lattice, scaled_positions, numbers = spglib.standardize_cell(
        cell, to_primitive=True, symprec=symprec
    )
    s = Structure(lattice, numbers, scaled_positions)
    standard_structure = s.get_sorted_structure()
    return standard_structure

def ibrav_to_cell(system):
    """
    Convert a value of ibrav to a cell. Any unspecified lattice dimension
    is set to 0.0, but will not necessarily raise an error. Also return the
    lattice parameter.

    Parameters
    ----------
    system : dict
        The &SYSTEM section of the input file, containing the 'ibrav' setting,
        and either celldm(1)..(6) or a, b, c, cosAB, cosAC, cosBC.

    Returns
    -------
    cell : Cell
        The cell as an ASE Cell object

    Raises
    ------
    KeyError
        Raise an error if any required keys are missing.
    NotImplementedError
        Only a limited number of ibrav settings can be parsed. An error
        is raised if the ibrav interpretation is not implemented.
    """
    if "celldm(1)" in system and "a" in system:
        raise KeyError("do not specify both celldm and a,b,c!")
    elif "celldm(1)" in system:
        # celldm(x) in bohr
        alat = au2ang(system["celldm(1)"])
        b_over_a = system.get("celldm(2)", 0.0)
        c_over_a = system.get("celldm(3)", 0.0)
        cosab = system.get("celldm(4)", 0.0)
        cosac = system.get("celldm(5)", 0.0)
        cosbc = 0.0
        if system["ibrav"] == 14:
            cosbc = system.get("celldm(4)", 0.0)
            cosac = system.get("celldm(5)", 0.0)
            cosab = system.get("celldm(6)", 0.0)
    elif "a" in system:
        # a, b, c, cosAB, cosAC, cosBC in Angstrom
        raise NotImplementedError(
            "params_to_cell() does not yet support A/B/C/cosAB/cosAC/cosBC"
        )
    else:
        raise KeyError("Missing celldm(1)")

    if system["ibrav"] == 1:
        cell = np.identity(3) * alat
    elif system["ibrav"] == 2:
        cell = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]]) * (
            alat / 2
        )
    elif system["ibrav"] == 3:
        cell = np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]]) * (
            alat / 2
        )
    elif system["ibrav"] == -3:
        cell = np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]) * (
            alat / 2
        )
    elif system["ibrav"] == 4:
        cell = (
            np.array([[1.0, 0.0, 0.0], [-0.5, 0.5 * 3**0.5, 0.0], [0.0, 0.0, c_over_a]])
            * alat
        )
    elif system["ibrav"] == 5:
        tx = ((1.0 - cosab) / 2.0) ** 0.5
        ty = ((1.0 - cosab) / 6.0) ** 0.5
        tz = ((1 + 2 * cosab) / 3.0) ** 0.5
        cell = np.array([[tx, -ty, tz], [0, 2 * ty, tz], [-tx, -ty, tz]]) * alat
    elif system["ibrav"] == -5:
        ty = ((1.0 - cosab) / 6.0) ** 0.5
        tz = ((1 + 2 * cosab) / 3.0) ** 0.5
        a_prime = alat / 3**0.5
        u = tz - 2 * 2**0.5 * ty
        v = tz + 2**0.5 * ty
        cell = np.array([[u, v, v], [v, u, v], [v, v, u]]) * a_prime
    elif system["ibrav"] == 6:
        cell = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, c_over_a]]) * alat
    elif system["ibrav"] == 7:
        cell = np.array(
            [[1.0, -1.0, c_over_a], [1.0, 1.0, c_over_a], [-1.0, -1.0, c_over_a]]
        ) * (alat / 2)
    elif system["ibrav"] == 8:
        cell = (
            np.array([[1.0, 0.0, 0.0], [0.0, b_over_a, 0.0], [0.0, 0.0, c_over_a]])
            * alat
        )
    elif system["ibrav"] == 9:
        cell = (
            np.array(
                [
                    [1.0 / 2.0, b_over_a / 2.0, 0.0],
                    [-1.0 / 2.0, b_over_a / 2.0, 0.0],
                    [0.0, 0.0, c_over_a],
                ]
            )
            * alat
        )
    elif system["ibrav"] == -9:
        cell = (
            np.array(
                [
                    [1.0 / 2.0, -b_over_a / 2.0, 0.0],
                    [1.0 / 2.0, b_over_a / 2.0, 0.0],
                    [0.0, 0.0, c_over_a],
                ]
            )
            * alat
        )
    elif system["ibrav"] == 10:
        cell = (
            np.array(
                [
                    [1.0 / 2.0, 0.0, c_over_a / 2.0],
                    [1.0 / 2.0, b_over_a / 2.0, 0.0],
                    [0.0, b_over_a / 2.0, c_over_a / 2.0],
                ]
            )
            * alat
        )
    elif system["ibrav"] == 11:
        cell = (
            np.array(
                [
                    [1.0 / 2.0, b_over_a / 2.0, c_over_a / 2.0],
                    [-1.0 / 2.0, b_over_a / 2.0, c_over_a / 2.0],
                    [-1.0 / 2.0, -b_over_a / 2.0, c_over_a / 2.0],
                ]
            )
            * alat
        )
    elif system["ibrav"] == 12:
        sinab = (1.0 - cosab**2) ** 0.5
        cell = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [b_over_a * cosab, b_over_a * sinab, 0.0],
                    [0.0, 0.0, c_over_a],
                ]
            )
            * alat
        )
    elif system["ibrav"] == -12:
        sinac = (1.0 - cosac**2) ** 0.5
        cell = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, b_over_a, 0.0],
                    [c_over_a * cosac, 0.0, c_over_a * sinac],
                ]
            )
            * alat
        )
    elif system["ibrav"] == 13:
        sinab = (1.0 - cosab**2) ** 0.5
        cell = (
            np.array(
                [
                    [1.0 / 2.0, 0.0, -c_over_a / 2.0],
                    [b_over_a * cosab, b_over_a * sinab, 0.0],
                    [1.0 / 2.0, 0.0, c_over_a / 2.0],
                ]
            )
            * alat
        )
    elif system["ibrav"] == 14:
        sinab = (1.0 - cosab**2) ** 0.5
        v3 = [
            c_over_a * cosac,
            c_over_a * (cosbc - cosac * cosab) / sinab,
            c_over_a
            * ((1 + 2 * cosbc * cosac * cosab - cosbc**2 - cosac**2 - cosab**2) ** 0.5)
            / sinab,
        ]
        cell = (
            np.array([[1.0, 0.0, 0.0], [b_over_a * cosab, b_over_a * sinab, 0.0], v3])
            * alat
        )
    else:
        raise NotImplementedError(
            "ibrav = {} is not implemented" "".format(system["ibrav"])
        )

    return np.array(cell)

def qe_get_ibrav(structure):
    """
    Transforms a structure into the QE standard format to make sure that the symmetry is detected correctly.
    INPUT:
        structure:      Structure that is analyzed
    OUTPUT:
        espresso_in:    Dictionary containing celldm and ibrav for the QE input file
        qe_structure:   Standardized structure
    """
    # first we determine the conventional structure
    sym = SpacegroupAnalyzer(structure, symprec=1e-5)
    std_struct = sym.get_conventional_standard_structure(international_monoclinic=True)

    # obtain the space group number
    spg = sym.get_space_group_number()

    # this is the structure in espresso input format
    espresso_in = {}
    espresso_in["celldm(1)"] = ang2au(std_struct.lattice.a)

    if spg in [
        195,
        198,
        200,
        201,
        205,
        207,
        208,
        212,
        213,
        215,
        218,
        221,
        222,
        223,
        224,
    ]:
        # simple cubic
        espresso_in["ibrav"] = 1
    elif spg in [196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228]:
        # face-centered cubic
        espresso_in["ibrav"] = 2
    elif spg in [197, 199, 204, 206, 211, 214, 217, 220, 229, 230]:
        espresso_in["ibrav"] = 3
    elif ((spg >= 168) and (spg <= 194)) or spg in [
        143,
        144,
        145,
        147,
        149,
        150,
        151,
        152,
        153,
        154,
        156,
        157,
        158,
        159,
        162,
        163,
        164,
        165,
    ]:
        # hexagonal
        espresso_in["ibrav"] = 4
        espresso_in["celldm(3)"] = std_struct.lattice.c / std_struct.lattice.a
    elif spg in [146, 148, 155, 160, 161, 166, 167]:
        # rhombohedral
        espresso_in["ibrav"] = 5
        aR = math.sqrt(std_struct.lattice.a**2 / 3 + std_struct.lattice.c**2 / 9)
        cosgamma = (2 * std_struct.lattice.c**2 - 3 * std_struct.lattice.a**2) / (
            2 * std_struct.lattice.c**2 + 6 * std_struct.lattice.a**2
        )
        espresso_in["celldm(1)"] = ang2au(aR)
        espresso_in["celldm(4)"] = cosgamma
    elif spg in [
        75,
        76,
        77,
        78,
        81,
        83,
        84,
        85,
        86,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
    ]:
        # simple Tetragonal
        espresso_in["ibrav"] = 6
        espresso_in["celldm(3)"] = std_struct.lattice.c / std_struct.lattice.a
    elif spg in [
        79,
        80,
        82,
        87,
        88,
        97,
        98,
        107,
        108,
        109,
        110,
        119,
        120,
        121,
        122,
        139,
        140,
        141,
        142,
    ]:
        # body-Centered Tetragonal
        espresso_in["ibrav"] = 7
        espresso_in["celldm(3)"] = std_struct.lattice.c / std_struct.lattice.a
    elif spg in [
        16,
        17,
        18,
        19,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
    ]:
        # simple Orthorhombic
        espresso_in["ibrav"] = 8
        espresso_in["celldm(2)"] = std_struct.lattice.b / std_struct.lattice.a
        espresso_in["celldm(3)"] = std_struct.lattice.c / std_struct.lattice.a
    elif spg in [20, 21, 35, 36, 37, 38, 39, 40, 41, 63, 64, 65, 66, 67, 68]:
        # base-Centered Orthorhombic
        espresso_in["ibrav"] = 9
        espresso_in["celldm(2)"] = std_struct.lattice.b / std_struct.lattice.a
        espresso_in["celldm(3)"] = std_struct.lattice.c / std_struct.lattice.a
    elif spg in [22, 42, 43, 69, 70]:
        # face-Centered Orthorhombic
        espresso_in["ibrav"] = 10
        espresso_in["celldm(2)"] = std_struct.lattice.b / std_struct.lattice.a
        espresso_in["celldm(3)"] = std_struct.lattice.c / std_struct.lattice.a
    elif spg in [23, 24, 44, 45, 46, 71, 72, 73, 74]:
        # body-Centered Orthorhombic
        espresso_in["ibrav"] = 11
        espresso_in["celldm(2)"] = std_struct.lattice.b / std_struct.lattice.a
        espresso_in["celldm(3)"] = std_struct.lattice.c / std_struct.lattice.a
    else:
        raise NotImplementedError(f"ibrav not defined for spg {spg}")

    # get espresso unit cell lattice
    cell = ibrav_to_cell(espresso_in)

    # get sites for the new lattice
    new_sites = []
    latt = Lattice(cell)
    for s in std_struct:
        new_s = PeriodicSite(
            s.specie,
            s.coords,
            latt,
            to_unit_cell=True,
            coords_are_cartesian=True,
            properties=s.properties,
        )
        # I set the tolerance to the position tolerance of the periodic site which is 1e-5.
        # The default tolerance for the is_periodic_image function is 1e-8 which sometimes
        # does not recognize that a set of sites is periodic.
        if not any(
            [
                new_s.is_periodic_image(ns, tolerance=new_s.position_atol)
                for ns in new_sites
            ]
        ):
            new_sites.append(new_s)

    # generate output structure
    qe_structure = Structure.from_sites(
        new_sites,
        to_unit_cell=True,
        validate_proximity=True, # validate_proximity=True should be set to be save ...
    )

    # check if the new and old primitive cell still match
    matcher = StructureMatcher(primitive_cell=False)
    if matcher.fit(structure, qe_structure):
        return espresso_in, qe_structure
    else:
        raise Exception(
            f"""
                QUITTING: The input primitive cell and the new primitive cell for Quantum Espresso dont match! 
                The detection of periodic sites may have done wrong! The calculation will new run with ibrav=0
                which may slow the calculation, as less symmetries may be detected!
                """
        )

def qe_init_structure(structure_load):
    """
    Initialize a structure stored earlier in a .pkl for further calculations.
    """

    # standardize the structure and get the ibrav
    structure = qe_standardize_cell(structure_load, symprec=1e-5)
    try:
        ibrav, structure = qe_get_ibrav(structure)
    except Exception:
        ibrav = 0

    return structure, ibrav

def qe_get_electrons(calc_data):
    """
    Gets the total number of included pseudopotential electrons in a structure saved in a calc_data.
    INPUT:
        calc_data:      calc_data class object containing the structure
    OUTPUT:
        electrons:      Number of electrons
    """
    electrons = 0
    for elem in calc_data.structure.species:
        with open(os.path.join(calc_data.pseudo, elem.name + ".upf")) as file:
            for line in file:
                if "z_valence" in line:
                    electrons += float(
                        re.findall(
                            "-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[+-]?\ *[0-9]+)?", line
                        )[0]
                    )
                    break

    return electrons

def qe_read_tot_energy(calc_data):
    """
    Read out total energy from the results of a QE calculation.
    INPUT:
        calc_data:      Class containing all necessary information
    OUTPUT:
        etot:           Total energy (Ha)
    """

    tree = ET.parse(f"out/{calc_data.id}.xml")
    etot = tree.find(".//{*}etot").text
    etot = float(etot)
    return etot

def qe_read_bandstructure(outdir_path):
    """
    Parses the eigenvalues as a array of size (num_kpt, num_bands).
    INPUT:
        id:             Materials Project ID
    OUTPUT:
        bs_matrix:      Array of eigenvalues for each k-point and band (Ha)
    """

    # parse the xml file
    tree = ET.parse(outdir_path)
    root = tree.getroot()

    # get the number of bands
    bands = root.iter("nbnd")
    for b in bands:
        nbnd = int(b.text)

    # parse the eigenvalues into a matrix
    eigenvalues = []
    for eigs in root.iter("eigenvalues"):
        eigs_num = [float(num) for num in eigs.text.strip("\n").split()]
        eigenvalues.append(eigs_num)
    eigenvalues = np.array(eigenvalues)
    bs_matrix = np.zeros([len(eigenvalues), nbnd])
    for i in range(len(eigenvalues)):
        bs_matrix[i, :] = eigenvalues[i]

    return bs_matrix

def qe_get_gw_band_range(num_elec, outdir_path, w_range=3):
    """
    Find the minimum number of bands needed to convergence the dielectric function 
    until w_range. This is not needed for the benchmark but still here...
    INPUT:
        num_elec:       Number of electrons in the system
        outdir_path:    Path to the QE output directory
        w_range:        All bands w_range eV up (down) from the VBM (CBM)
    OUTPUT:
        vb_start:       Lowest VB for which we calculate the GW correction
        cb_end:         Highest CB for which we calculate the GW correction
    """
    # convert the w_range to Ha
    w_range = ev2ha(w_range)

    # parse the band structure
    bs_matrix = qe_read_bandstructure(outdir_path)

    # index and energy of the vbm and cbm
    vbm_idx = int(num_elec / 2) - 1
    vbm = np.max(bs_matrix[:, vbm_idx])
    cbm_idx = vbm_idx + 1
    cbm = np.min(bs_matrix[:, cbm_idx])

    # find the first band which is more than w_max away from the Fermi energy
    # for this we check that the lower edges of the conduction band is more
    # than w_max away from the Fermi energy
    vb_start = np.argmax(np.min(vbm - bs_matrix, axis=0) > w_range) + 1
    cb_end = np.argmax(np.min(bs_matrix - cbm, axis=0) > w_range) + 1

    return [vb_start, cb_end]

def bg_get_bandgap(calc_data):
    """
    Calculates band gap from the results of a QE calculation.
    INPUT:
        calc_data:      Class containing all necessary information
    OUTPUT:
        egap:           Band gap (Ha)
    """

    root = ET.parse(f"out/{calc_data.id}.xml").getroot()

    kpoints = []
    for type_tag in root.iter("k_point"):
        kpoint = re.findall(r"[-+]?\d+.\d+e-?\d+", type_tag.text)
        for i in range(len(kpoint)):
            kpoint[i] = float(kpoint[i])
        kpoints.append(kpoint)

    evalues = []
    for type_tag in root.iter("eigenvalues"):
        value = re.findall(r"[-+]?\d+.\d+e-?\d+", type_tag.text)
        for i in range(len(value)):
            value[i] = float(value[i])
        evalues.append(value)

    for type_tag in root.iter("nelec"):
        value = re.findall(r"[-+]?\d+.\d+e-?\d+", type_tag.text)
    nelec = int(float(value[0]))

    vbm = int(nelec / 2 - 1)
    cbm = vbm + 1

    evalues = np.array(evalues).reshape((len(kpoints), len(evalues[0])))

    egap = float(ha2ev(np.min(evalues[:, cbm]) - np.max(evalues[:, vbm])))

    return egap

def read_num_electrons(calc_data):
    """
    Reads the number of electons from the XML output of QE.

    INPUT:
        calc_data:       Class containing all necessary information
    OUTPUT:
        nelec:           Number of electrons
    """

    root = ET.parse(f"out/{calc_data.id}.xml").getroot()

    for type_tag in root.iter("nelec"):
        nelec = (re.findall(r"\d+.\d+e\d+", type_tag.text))[0]

    return float(nelec)
"""
The basic class for calculations.
"""

# local imports
from src.utils.basic_utils import get_kpt_grid

class calc_data:
    """
    Class which contains data about calculation in code-agnostic version
    structure:      pymatgen structure class object
    id:             Database identifier for the system (e.g. mp-8, mp-1234, ...)
    ibrav:          QE input string to detect all symmetries
    calc_type:      Calculation type like scf, nscf, bands
    kppa:           k-grid density
    pw_cutoff:      Plane wave energy cutoff
    pseudo:         Path to the pseudo potential directory
    degauss:        Smearing parameter, important for the scf convergence
    """

    def __init__(
        self,
        structure,
        id=0,
        ibrav=0,
        calc_type="scf",
        kppa=1500,
        pw_cutoff=-1,
        pseudo="",
        degauss=0.00735,
    ):
        self.structure = structure
        self.id = id
        self.identifier = id
        self.ibrav = ibrav
        self.calc_type = calc_type
        self.kppa = kppa
        self.pseudo = pseudo
        self.degauss = degauss

        # if no cutoff is supplied, use the default value
        if pw_cutoff == -1:
            self.pw_cutoff = 60 # Ry (default value for the SG15 pseudo potentials)
        else:
            self.pw_cutoff = pw_cutoff

        # we currently only support scf and nscf calculations
        if calc_type in ["scf", "nscf"]:
            self.k_points_grid = get_kpt_grid(structure, kppa)
            self.prefix = ( # prefix not necessary for all calculations
                self.identifier
                + "_"
                + self.calc_type
                + "_k"
                + str(self.kppa)
                + "_E"
                + str(self.pw_cutoff)
            )
        else:
            raise Exception(
                f"""QUITTING: Calculation type {calc_type} not implemented!"""
            )

"""
Functions that just write input files for QE.
"""

# external imports
import numpy as np

# local imports
import src.utils.qe_helper as qe_helper

def write_pw(
    calc_data,
    occupations="smearing",
    disk_io="high",
    input_dft="pbe",
    noinv=False,
    nosym=False,
    diagonalization="david",
    kpoints_mode="automatic",
    shift=[0, 0, 0],
    n_bands=0,
    output_filename=None,
):
    """
    Writes the input file for a pw.x calculation, optimized for metals.
    It is a bit more suffisticated than the standard pymatgen implementation.
    INPUT:
        calc_data:          calc_data class object of the calculation one wants to run
        For the other inputs, see the pw.x documentation
    OUTPUT:
        output_filename:    Name of the written input file
    """

    if output_filename is None:
        output_filename = (
            "pw_"
            + calc_data.identifier
            + "_"
            + calc_data.calc_type
            + "_k"
            + str(calc_data.kppa)
            + "_E"
            + str(round(calc_data.pw_cutoff)) # cutoff should be an integer
        )
    else:
        output_filename = output_filename
    pseudo = {}
    for elem in calc_data.structure.types_of_species:
        pseudo[elem.name] = elem.name + ".upf"
    k_points_grid = calc_data.k_points_grid
    control = {
        "calculation": calc_data.calc_type,
        "prefix": calc_data.identifier,
        "outdir": "out",
        "pseudo_dir": calc_data.pseudo,
        "disk_io": disk_io,
    }
    system = {
        "input_dft": input_dft,
        "ecutwfc": calc_data.pw_cutoff,
        "occupations": occupations,
        "degauss": calc_data.degauss,
        "smearing": "gaussian",
        "noinv": noinv,
        "nosym": nosym,
    }
    if n_bands == 0:
        system.update(
            {
                "nbnd": int(
                    np.ceil(
                        np.max(
                            [
                                qe_helper.qe_get_electrons(calc_data),
                                8,
                            ]
                        )
                    )
                ),
            }
        )
    else:
        system.update({"nbnd": n_bands})
    if calc_data.ibrav != 0:
        system.update(calc_data.ibrav)
    if calc_data.calc_type == "nscf": # important for Yambo
        system.update({"force_symmorphic": True})
    electrons = {
        "conv_thr": 1e-10,
        "electron_maxstep": 200,
        "diagonalization": diagonalization,
    }
    electrons.update({"mixing_beta": 0.6})
    if calc_data.calc_type == "nscf":
        electrons.update({"diago_full_acc": True})
        electrons.update({"diago_thr_init": 5e-6})
    input = qe_helper.qe_PWInput(
        calc_data.structure,
        pseudo=pseudo,
        kpoints_grid=k_points_grid,
        control=control,
        system=system,
        electrons=electrons,
        kpoints_mode=kpoints_mode,
        kpoints_shift=shift,
    )
    input.write_file(output_filename + ".in")

    return output_filename

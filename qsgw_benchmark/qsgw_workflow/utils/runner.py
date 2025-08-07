"""
Contains all functions that perform calculations.
"""

# external imports
import os
import re
import sys
import numpy as np

# internal imports
import qsgw_workflow.utils.helper as helper
from qsgw_workflow.utils.system_config import execute_command

def dft_kpt_conv(struct, kppa, ncores, max_iter=20, etol=1e-4):
    """
    Converges a DFT calculation with respect to the k-grid.
    Reference: https://www.questaal.org/tutorial/lmf/basis_set/#4-gmax-and-nkabc (4.2)
    INPUT:
        struct:         pymatgen structure object
        kppa:           Initial k-point density
        ncores:         Number of cores to be used for the calculation
        max_iter:       Maximum number of iterations
        etol:           Tolerance in the Harris-Foulkes energy (Ry/atom)
    OUTPUT:
        kppa:           Converged k-point density
        kpts:           Converged k-grid as a list
    """
    print("\nStarting DFT k-grid convergence:", flush=True)
    print(
        f"    Tolerance = {etol:.1E} Ry/atom ({etol*len(struct.sites):.1E} Ry)",
        flush=True,
    )
    conv_data = []
    # initial calculation
    iter = 1
    fname = f"dft_kpt_{iter:d}.log"
    kpts = helper.get_kpt_grid(struct, kppa)
    kstr = f"-vnk1={kpts[0]:d} -vnk2={kpts[1]:d} -vnk3={kpts[2]:d}"
    # here we add "--pr=55" to obtain the orbital order of the hamiltonian, which can be good to know
    execute_command(
        f"mpirun -np {ncores:d} lmf {kstr:s} --pr55 --quit=band ctrl.mat > {fname:s}"
    )
    ehf = helper.grep_ehf(fname)
    conv_data += [[iter, kppa, kpts, ehf]]
    print_kpt = f"{kpts[0]:d}x{kpts[1]:d}x{kpts[2]:d}"
    print(
        f"    {iter:<2d} | kppa = {kppa:<6d} | kpts = {print_kpt:<8s} | ehf = {ehf:>5.6f} Ry",
        flush=True,
    )
    # next step
    iter += 1
    fname = f"dft_kpt_{iter:d}.log"
    kppa = helper.increase_kppa(struct, kppa)
    kpts = helper.get_kpt_grid(struct, kppa)
    kstr = f"-vnk1={kpts[0]:d} -vnk2={kpts[1]:d} -vnk3={kpts[2]:d}"
    execute_command(
        f"mpirun -np {ncores:d} lmf {kstr:s} --quit=band ctrl.mat > {fname:s}"
    )
    ehf = helper.grep_ehf(fname)
    conv_data += [[iter, kppa, kpts, ehf]]
    print_kpt = f"{kpts[0]:d}x{kpts[1]:d}x{kpts[2]:d}"
    print(
        f"    {iter:<2d} | kppa = {kppa:<6d} | kpts = {print_kpt:<8s} | ehf = {ehf:>5.6f} Ry",
        flush=True,
    )
    # convergence loop
    while np.abs(conv_data[-2][3] - conv_data[-1][3]) > etol * len(struct.sites):
        iter += 1
        if iter == max_iter + 1:
            open("dft_kpt_conv_max_iter.txt", "w").close
            sys.exit("\nReached the maximum number of iterations!")
        fname = f"dft_kpt_{iter:d}.log"
        kppa = helper.increase_kppa(struct, kppa)
        kpts = helper.get_kpt_grid(struct, kppa)
        kstr = f"-vnk1={kpts[0]:d} -vnk2={kpts[1]:d} -vnk3={kpts[2]:d}"
        execute_command(
            f"mpirun -np {ncores:d} lmf {kstr:s} --quit=band ctrl.mat > {fname:s}"
        )
        ehf = helper.grep_ehf(fname)
        conv_data += [[iter, kppa, kpts, ehf]]
        print_kpt = f"{kpts[0]:d}x{kpts[1]:d}x{kpts[2]:d}"
        print(
            f"    {iter:<2d} | kppa = {kppa:<6d} | kpts = {print_kpt:<8s} | ehf = {ehf:>5.6f} Ry",
            flush=True,
        )
    print("Finished the DFT k-grid convergence.", flush=True)
    return kppa, kpts, conv_data

def run_dft(ncores):
    """
    Run a self-consistent DFT calculation.
    INPUT:
        ncores:         Number of cores to be used for the calculation
    """
    print("\nRunning a self-consistent DFT calculation.", flush=True)
    execute_command(f"rm -f mixm.mat rst.mat") # clean restart...
    execute_command(f"mpirun -np {ncores:d} lmf ctrl.mat > dft.log")
    
def run_dft_with_soc_post_gw(ncores):
    """
    Run a self-consistent DFT calculation that includes spin-orbit coupling.
    This function is optimized to run as a post-processing step after a QSGW or QSGW^ calculation.
    The directory will be cleaned up as if nothing happened. The band gap can be read from the 'dft_with_soc.log' file.
    INPUT:
        ncores:         Number of cores to be used for the calculation
    """
    print("\nRunning a self-consistent DFT calculation with spin-orbit coupling.", flush=True)
    if not os.path.exists("sigm.mat") and not os.path.exists("sigm2.mat"):
        sys.exit("DFT+SOC is not currently supported without a prior QSGW or QSGW^.")
    execute_command("cp sigm.mat sigm_tmp.mat; cp sigm2.mat sigm2_tmp.mat; cp rst.mat rst_tmp.mat; cp mixm.mat mixm_tmp.mat")
    execute_command("lmf ctrl.mat --wsig:fbz > dft_soc.log")
    execute_command("cp sigm2.mat sigm.mat")
    execute_command(f"mpirun -np {ncores:d} lmf -vso=1 ctrl.mat > dft_soc.log")
    execute_command("rm mixm.mat rst.mat sigm.mat sigm2.mat")
    execute_command("mv sigm_tmp.mat sigm.mat; mv sigm2_tmp.mat sigm2.mat; mv rst_tmp.mat rst.mat; mv mixm_tmp.mat mixm.mat")

def get_bandstructure(label):
    """
    This function calculates and parses the band structure on a standardized symmetry path.
    INPUT:
        label:          Tag for the band structure file, e.g., "lda", "qsgw", "qsgwbar", ...
    OUTPUT:
        bs:             Dictionary with all information about the band structure
    """
    print(f"\nCalculating the {label.upper():s} band structure.", flush=True)
    execute_command(f"lmchk --syml ctrl.mat > bs_dump.log")
    execute_command(f"lmf --band~fn=syml ctrl.mat >> bs_dump.log")
    execute_command(f"mv bnds.mat bnds_{label.lower():s}.mat")
    bs = helper.parse_bandstructure(f"bnds_{label.lower():s}.mat")
    bs["elem_list"] = None # compatibility with other band structure functions
    return bs

def get_atom_proj_bandstructure(struct, label):
    """
    This function calculates and parses the band structure on a standardized symmetry path.
    Projections on unique elements in the unit cell are obtained. Due to the limitations of 
    the code, we can only project to a maximum of three elements. If there are more than three 
    unique elements in a structure, we take the three with the lowest atomic number Z.
    INPUT:
        struct:         pymatgen structure object
        label:          Tag for the band structure file (e.h. "lda", "qsgw", "qsgwbar", ...)
    OUTPUT: 
        bs:             Dictionary with all information about the band structure
    """
    elem_list = helper.get_unique_atoms(struct)
    print(
        f"\nCalculating the site-projected {label.upper():s} band structure.",
        flush=True,
    )
    proj_str = "--band"
    if len(elem_list) == 1:
        Z = elem_list[0][1]
        print(
            f"Obtaining site-resolved projections for {elem_list[0][0]:s}.", flush=True
        )
        proj_str += f"~scol@z={Z:d}"
    elif len(elem_list) == 2:
        Z1 = elem_list[0][1]
        Z2 = elem_list[1][1]
        print(
            f"Obtaining site-resolved projections for {elem_list[0][0]:s} and {elem_list[1][0]:s}.",
            flush=True,
        )
        proj_str += f"~scol@z={Z1:d}~scol2@z={Z2:d}"
    elif len(elem_list) == 3:
        Z1 = elem_list[0][1]
        Z2 = elem_list[1][1]
        Z3 = elem_list[2][1]
        print(
            f"Obtaining site-resolved projections for {elem_list[0][0]:s}, {elem_list[1][0]:s} and {elem_list[2][0]:s}.",
            flush=True,
        )
        proj_str += f"~scol@z={Z1:d}~scol2@z={Z2:d}~scol3@z={Z3:d}"
    if len(elem_list) > 3:
        print("Omitting site projections for:", flush=True)
        for i in range(3, len(elem_list)):
            print(f"        {elem_list[i][0]:s}", flush=True)
    proj_str += "~fn=syml"
    # print(proj_str, flush=True) # debugging
    execute_command(f"lmchk --syml ctrl.mat > bs_{label.lower():s}.log")
    execute_command(f"lmf {proj_str:s} ctrl.mat >> bs_{label.lower():s}.log")
    execute_command(f"mv bnds.mat bnds_{label.lower():s}.mat")
    bs = helper.parse_bandstructure(f"bnds_{label.lower():s}.mat")
    bs["elem_list"] = elem_list # list of [elem, Z], sorted by Z, for which the projections are obtained
    return bs

def get_pdos(label):
    """
    This function obtains the total DOS and PDOS.
    Projections up to "lcut=2", i.e., the d-orbitals, are obtained for all sites.
    INPUT:
        label:              Tag for the PDOS file (e.h. "lda", "qsgw", "qsgwbar", ...)
    OUTPUT
        pdos:               Dictionary with all information about the PDOS
    """
    pdos_str = "--pdos~mode=1~lcut=2" # resolved by l and site (up to the d-orbitals)
    dos_str = f"--dos~npts=4001~window=-2,2~ef0" # https://www.questaal.org/docs/input/commandline/#dos
    print(f"\nCalculating the {label.upper():s} DOS and PDOS.", flush=True)
    execute_command(f"lmf {pdos_str:s} --quit=rho ctrl.mat > dos_{label.lower():s}.log")
    execute_command(f"lmdos {pdos_str:s} {dos_str:s} ctrl.mat >> dos_{label.lower():s}.log")
    execute_command(f"mv dos.mat dos_{label.lower():s}.mat")
    pdos = helper.parse_pdos(f"dos_{label.lower():s}.mat")
    with open(f"dos_{label.lower():s}.log") as f:
        dos_str = f.read()
    pattern = r"^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\d+:\d+)$"
    matches = re.findall(pattern, dos_str, re.MULTILINE)
    pdos_helper_str = "Element  Index Range (one-based)\n"
    for i, match in enumerate(matches):
        if i == len(matches) - 1:
            pdos_helper_str += f"{match[2]:<7s}  {match[3]}"
        else:
            pdos_helper_str += f"{match[2]:<7s}  {match[3]}\n"
    # string explaining the channels (first dimension) of the PDOS array
    print(pdos_helper_str, flush=True)
    pdos.update({"pdos_helper_str": pdos_helper_str})
    return pdos

def eps_kpt_conv(struct, kppa, ncores, sc_tol=0.95, max_iter=30):
    """
    This function converges the IPA dielectric tensor with respect to the k-grid.
    The converged IPA dielectric tensor is also given as output.
    INPUT:
        struct:         pymatgen structure object
        kppa:           Initial k-point density
        ncores:         Number of cores to be used for the calculation
        sctol:          Tolerance for the similarity coefficient
        max_iter:       Maximum number of iterations
    OUTPUT:
        kppa:           Converged k-point density
        kpts:           Converged k-grid as a list
        eps:            Converged dielectric tensor
    """
    print("\nStarting the IPA dielectric tensor k-grid convergence:", flush=True)
    print(f"    Tolerance: SC > {sc_tol:.2f}", flush=True)
    # key for the output dictionary and tag for the output file
    conv_data = []
    # initial calculation
    iter = 1
    fname = f"eps_kpt_{iter:d}.log"
    kpts = helper.get_kpt_grid(struct, kppa)
    kstr = f"-vnk1={kpts[0]:d} -vnk2={kpts[1]:d} -vnk3={kpts[2]:d}"
    execute_command(
        f"mpirun -np {ncores:d} lmf {kstr:s} -vloptic=1 --quit=rho ctrl.mat > {fname:s}"
    )
    eps = helper.parse_eps("opt.mat")
    conv_data += [[iter, kppa, kpts, eps, 0]] # 0 is the SC
    print_kpt = f"{kpts[0]:d}x{kpts[1]:d}x{kpts[2]:d}"
    print(
        f"    {iter:<2d} | kppa = {kppa:<6d} | kpts = {print_kpt:<8s} | SC = NaN",
        flush=True,
    )
    # next step
    iter += 1
    fname = f"eps_kpt_{iter:d}.log"
    kppa = helper.increase_kppa(struct, kppa)
    kpts = helper.get_kpt_grid(struct, kppa)
    kstr = f"-vnk1={kpts[0]:d} -vnk2={kpts[1]:d} -vnk3={kpts[2]:d}"
    execute_command(
        f"mpirun -np {ncores:d} lmf {kstr:s} -vloptic=1 --quit=rho ctrl.mat > {fname:s}"
    )
    eps = helper.parse_eps("opt.mat")
    sc = helper.calc_sc(conv_data[-1][3], eps)
    conv_data += [[iter, kppa, kpts, eps, sc]]
    print_kpt = f"{kpts[0]:d}x{kpts[1]:d}x{kpts[2]:d}"
    print(
        f"    {iter:<2d} | kppa = {kppa:<6d} | kpts = {print_kpt:<8s} | SC = {sc:<.3f}",
        flush=True,
    )
    # convergence loop
    while conv_data[-1][4] < sc_tol:
        iter += 1
        if iter == max_iter + 1:
            open("eps_kpt_conv_max_iter.txt", "w").close
            sys.exit("\nReached the maximum number of iterations!")
        fname = f"eps_kpt_{iter:d}.log"
        kppa = helper.increase_kppa(struct, kppa)
        kpts = helper.get_kpt_grid(struct, kppa)
        kstr = f"-vnk1={kpts[0]:d} -vnk2={kpts[1]:d} -vnk3={kpts[2]:d}"
        execute_command(
            f"mpirun -np {ncores:d} lmf {kstr:s} -vloptic=1 --quit=rho ctrl.mat > {fname:s}"
        )
        eps = helper.parse_eps("opt.mat")
        sc = helper.calc_sc(conv_data[-1][3], eps)
        conv_data += [[iter, kppa, kpts, eps, sc]]
        print_kpt = f"{kpts[0]:d}x{kpts[1]:d}x{kpts[2]:d}"
        print(
            f"    {iter:<2d} | kppa = {kppa:<6d} | kpts = {print_kpt:<8s} | SC = {sc:<.3f}",
            flush=True,
        )
    print("Finished the IPA dielectric tensor k-grid convergence.", flush=True)
    execute_command(f"mv opt.mat opt_lda.mat")
    return kppa, kpts, eps, conv_data

def calc_eps(kpts, ncores, label):
    """
    This function calculates and parses the IPA dielectric tensor with respect to the k-grid.
    INPUT:
        kpts:           Converged k-grid as a list
        ncores:         Number of cores to be used for the calculation
        label:          Key for the output dictionary and tag for the output file
    OUTPUT:
        eps:            Converged dielectric tensor
    """
    print(f"\nCalculating the {label.upper():s} IPA dielectric tensor.", flush=True)
    kstr = f"-vnk1={kpts[0]:d} -vnk2={kpts[1]:d} -vnk3={kpts[2]:d}"
    print(f"(k-grid input string: {kstr:s})", flush=True) # debugging
    execute_command(
        f"mpirun -np {ncores:d} lmf {kstr:s} -vloptic=1 --quit=rho ctrl.mat > eps_{label.lower():s}.log"
    )
    execute_command(f"mv opt.mat opt_{label.lower():s}.mat")
    eps = helper.parse_eps(f"opt_{label.lower():s}.mat")
    return eps

def run_qsgw_with_error_handling(name, nnodes, ncores, misc_str, fname):
    """
    Function to run a QSGW calculation while handling errors.
    At the moment we only catch the inexact inverse Bloch transform error.
    INPUT:
        name:           Name of the material 
        nnodes:         Number of nodes to be used for the calculation
        ncores:         Number of cores to be used for the calculation
        misc_str:       Input switches for 'lmgw.sh'
        fname:          Name of the log file
    """
    counter = 0
    bloch_sum_error_flag = 1 # all error flags in our code are 1 if an error occurred and 0 if everything worked fine
    while bloch_sum_error_flag:
        if counter == 3:
            open("bloch_sum_error.txt", "w").close
            sys.exit(f"Inexact inverse Bloch transform error. Tried to increase RSRNGE twice!")
        helper.clean_qsgw(name, rst_flag=True, indent=True)
        mpi_str = helper.create_pqmap(nnodes, ncores)
        print("    Starting a QSGW calculation.", flush=True)
        execute_command(f"lmgw.sh {misc_str:s} {mpi_str:s} ctrl.mat > {fname:s}")
        bloch_sum_error_flag = helper.check_and_fix_bloch_sum()
        counter += 1

def qsgw_kpt_conv_semi(
    name, 
    struct, 
    init_kppa, 
    max_kppa,
    nnodes, 
    ncores,
    max_iter=10,
    etol=0.00184, # Ry (~25 meV)
):
    """
    Converge the QSGW self energy with respect to the k-grid.
    If the QSGW self energy k-grid is equal to the DFT k-grid we just stop.
    In the community, there is a common belief that self energy converges 
    faster than a regular DFT with respect to the k-grid.
    We effectively just do one-shot QSGW (QPG0W0) calculations.
    Reference: https://www.questaal.org/tutorial/gw/qsgw_fe/#1-k-convergence
    INPUT:
        name:           Name of the material
        struct:         pymatgen structure object
        init_kppa:      Initial k-point density
        max_kppa:       Maximum k-point density (converged DFT k-grid)
        nnodes:         Number of nodes to be used for the calculation
        ncores:         Number of cores to be used for the calculation
        max_iter:       Maximum number of iterations
        etol:           Tolerance in the band edges (Ry)
    OUTPUT:
        kppa:           Converged self energy k-point density
        kpts:           Converged self energy k-grid as a list
        error_flag:     Flag indicating if the k-grid convergence was not achieved when reaching the DFT k-grid
        conv_data:      Summary of the convergence steps
    """
    print("\nStarting QSGW k-grid convergence:", flush=True)
    print(
        "    We just do one-shot QSGW (QPG0W0) calculations and converge the band gap.",
        flush=True,
    )
    print(f"    Tolerance = {etol:.1E} Ry ({helper.ry2ev(etol):.1E} eV)", flush=True)
    etol = helper.ry2ev(etol) # 'helper.get_gap()' returns the band gap in eV
    conv_data = []
    # we just do a single qsgw iteration for the k-grid convergence of self energy
    misc_str = "--incount --sym --tol=1e-5 --split-w0 --maxit 0"
    # initial calculation
    kppa = init_kppa
    iter = 1
    fname = f"qsgw_kpt_{iter:d}.log"
    kpts = helper.get_kpt_grid(struct, kppa)
    helper.set_gw_kgrid(kpts)
    run_qsgw_with_error_handling(name, nnodes, ncores, misc_str, fname)
    gap = helper.get_gap("llmf")
    conv_data += [[iter, kppa, kpts, gap]]
    print_kpt = f"{kpts[0]:d}x{kpts[1]:d}x{kpts[2]:d}"
    if gap == -1:
        print(
            f"    {iter:<2d} | kppa = {kppa:<6d} | kpts = {print_kpt:<8s} | Interpolation of the self energy is erroneous.",
            flush=True,
        )
    else:
        print(
            f"    {iter:<2d} | kppa = {kppa:<6d} | kpts = {print_kpt:<8s} | gap = {gap:>5.6f} eV",
            flush=True,
        )
    # next step
    iter += 1
    fname = f"qsgw_kpt_{iter:d}.log"
    kppa = helper.increase_kppa(struct, kppa)
    if kppa > max_kppa:
        sys.exit("\nIncrease 'dft_kppa'!")
    kpts = helper.get_kpt_grid(struct, kppa)
    helper.set_gw_kgrid(kpts)
    run_qsgw_with_error_handling(name, nnodes, ncores, misc_str, fname)
    gap = helper.get_gap("llmf")
    conv_data += [[iter, kppa, kpts, gap]]
    print_kpt = f"{kpts[0]:d}x{kpts[1]:d}x{kpts[2]:d}"
    if gap == -1:
        print(
            f"    {iter:<2d} | kppa = {kppa:<6d} | kpts = {print_kpt:<8s} | Interpolation of the self energy is erroneous.",
            flush=True,
        )
    else:
        print(
            f"    {iter:<2d} | kppa = {kppa:<6d} | kpts = {print_kpt:<8s} | gap = {gap:>5.6f} eV",
            flush=True,
        )
    # convergence loop
    run_conv_flag = (np.abs(conv_data[-2][3] - conv_data[-1][3]) > etol) or (gap == -1)
    while run_conv_flag:
        iter += 1
        if iter == max_iter + 1:
            open("qsgw_kpt_conv_max_iter.txt", "w").close
            sys.exit("\nReached the maximum number of iterations!")
        fname = f"qsgw_kpt_{iter:d}.log"
        old_kppa = kppa
        kppa = helper.increase_kppa(struct, kppa)
        if kppa > max_kppa:
            print(
                "    The QSGW self energy k-grid is equal to the DFT k-grid. Stopping convergence.",
                flush=True,
            )
            error_flag = 1 # just so we know if the qsgw has converged to the k-grid or if we have reached the DFT k-grid
            return old_kppa, kpts, error_flag, conv_data
        kpts = helper.get_kpt_grid(struct, kppa)
        helper.set_gw_kgrid(kpts)
        run_qsgw_with_error_handling(name, nnodes, ncores, misc_str, fname)
        gap = helper.get_gap("llmf")
        conv_data += [[iter, kppa, kpts, gap]]
        print_kpt = f"{kpts[0]:d}x{kpts[1]:d}x{kpts[2]:d}"
        if gap == -1:
            print(
                f"    {iter:<2d} | kppa = {kppa:<6d} | kpts = {print_kpt:<8s} | Interpolation of the self energy is erroneous.",
                flush=True,
            )
            run_conv_flag = True
        else:
            print(
                f"    {iter:<2d} | kppa = {kppa:<6d} | kpts = {print_kpt:<8s} | gap = {gap:>5.6f} eV",
                flush=True,
            )
            run_conv_flag = (np.abs(conv_data[-2][3] - conv_data[-1][3]) > etol) or (
                gap == -1
            )
    print("Finished the QSGW k-grid convergence.", flush=True)
    error_flag = 0 
    return kppa, kpts, error_flag, conv_data

def run_qsgw(name, nnodes, ncores, max_iter=25, fresh_start_flag=True):
    """
    Performs a fully self-consistent QSGW calculation.
    Here we DO NOT clean up the directory before starting the calculation. 
    Normally this function is used right after the self energy k-grid convergence (see function above). 
    Collects the output data and checks if the self-consistency cycle has converged.
    If the calculation crashed, this function simply returns 'None'.
    INPUT:
        name:               Name of the material
        nnodes:             Number of nodes to be used for the calculation
        ncores:             Number of cores to be used for the calculation
        max_iter:           Maximum number of QSGW iteration
        fresh_start_flag:   Changes how the output is parsed of the self-consistent QSGW
                            is parsed, depending on whether we start at the 0th or 1st iteration
    OUTPUT:
        'None' if the calculation crashed...
        OR
        scf_data:           Summary of the self-consistency cycle
                            (list of lists, i.e., [[iteration, rms change in the self energy, band gap], [...]])
        error_flag:         Flag indicating if the self-consistency cycle did not converge
    """
    print("\nRunning a fully self-consistent QSGW calculation.", flush=True)
    print(f"    Tolerance = 1E-5 Ry", flush=True)
    # parallelization
    mpi_str = helper.create_pqmap(nnodes, ncores)
    """
    Run a full self-consistent QSGW:
        Three things can happen at this stage:
        1. Everything works and there is no inexact inverse Bloch transform error.
        2. The error occurs before the RMS tolerance is reached, then we just increase RSRNGE and run 'lmgw.sh' again.
        3. The error happens at the end of the computation after the RMS tolerance is reached. 
           A simple restart is not possible because the tolerance has been reached.
        -> There are probably smarter solutions, but if 2. or 3. happens, we just delete the directory and run the full QSGW again.
        (Maybe one can delete everything but the self energy and restart, but then one would have to reconstruct the iterations depending on where the error happend.)
    """
    misc_str = f"--incount --sym --tol=1e-5 --split-w0 --maxit={max_iter:d}"
    print("    Starting a fully self-consistent QSGW calculation.", flush=True)
    execute_command(f"lmgw.sh {misc_str:s} {mpi_str:s} ctrl.mat > qsgw.log")
    bloch_sum_error_flag = helper.check_and_fix_bloch_sum() # all error flags in our code are 1 if an error occurred and 0 if everything worked fine
    if bloch_sum_error_flag:
        helper.clean_qsgw(name, rst_flag=True, indent=True)
        mpi_str = helper.create_pqmap(nnodes, ncores)
        print(f"    Restarting a fully self-consistent QSGW calculation (using a higher RSRNGE).")
        execute_command(f"lmgw.sh {misc_str:s} {mpi_str:s} ctrl.mat > qsgw.log")
        bloch_sum_error_flag = helper.check_and_fix_bloch_sum()
        if bloch_sum_error_flag:
            open("bloch_sum_error.txt", "w").close
            sys.exit(f"Inexact inverse Bloch transform error. Already tried to increase RSRNGE!")
        fresh_start_flag = True
    # check that the QSGW actually finished
    with open("qsgw.log", "r") as f:
        last_line = f.readlines()[-1]
    if not re.match(r"lmgw: iter \d+ of \d+ completed in", last_line): 
        print("The QSGW has crashed!", flush=True)
        return None
    # retrieve information about the self-consistency cycle
    if fresh_start_flag:
        qsqw_rms_info = os.popen("grep RMS qsgw.log").read()
        iter_matches = re.findall(r"iter \d+", qsqw_rms_info)
        iter = [int(re.findall(r"\d+", x)[0]) for x in iter_matches]
        rms_matches = re.findall(r"sigma = [-+]?\d*\.?\d+e[-+]?\d+", qsqw_rms_info)
        rms = [float(re.findall(r"[-+]?\d*\.?\d+e[-+]?\d+", x)[0]) for x in rms_matches]
    else:
        # information from the first iteration (k-grid convergence)
        last_qsgw_kpt_log_idx = max([int(re.findall(r"\d+", f)[0]) for f in os.listdir() if "qsgw_kpt_" in f])
        qsqw_log_info = os.popen(f"grep RMS qsgw_kpt_{last_qsgw_kpt_log_idx:d}.log").read()
        iter_matches = re.findall(r"iter \d+", qsqw_log_info)
        iter = [int(re.findall(r"\d+", x)[0]) for x in iter_matches]
        rms_matches = re.findall(r"sigma = [-+]?\d*\.?\d+e[-+]?\d+", qsqw_log_info)
        rms = [float(re.findall(r"[-+]?\d*\.?\d+e[-+]?\d+", x)[0]) for x in rms_matches]
        # information about all other iterations (self-consistency)
        qsqw_rms_info = os.popen("grep RMS qsgw.log").read()
        iter_matches = re.findall(r"iter \d+", qsqw_rms_info)
        iter += [int(re.findall(r"\d+", x)[0]) for x in iter_matches]
        rms_matches = re.findall(r"sigma = [-+]?\d*\.?\d+e[-+]?\d+", qsqw_rms_info)
        rms += [float(re.findall(r"[-+]?\d*\.?\d+e[-+]?\d+", x)[0]) for x in rms_matches]
    # get the band gap evolution from the QSGW calculation
    run_dirs = [d for d in os.listdir() if "run" in d and os.path.isdir(d)]
    run_dirs.sort(key=lambda x: int(x.split("run")[0]))
    gaps = []
    for d in run_dirs:
        os.chdir(d)
        gaps += [helper.get_gap("llmf")]
        os.chdir("../")
    # gather everything in one array
    scf_data = np.column_stack([iter, rms, gaps])
    # log message
    print("    Iteration | RMS Sigma (Ry) | Band gap (eV)", flush=True)
    for i in range(len(iter)):
        print(f"    {iter[i]:<9d} | {rms[i]:<14.2E} | {gaps[i]:<2.3f}", flush=True)
    # convergence check
    if rms[-1] > 1e-5:
        print(
            "Full convergence of the QSGW self-consistency cycle was not achieved.",
            flush=True,
        )
        error_flag = 1
    else:
        error_flag = 0
    return scf_data, error_flag

def run_qsgw_with_bse(name, nnodes, ncores, max_iter=25):
    """
    Performs a fully self-consistent QSGW calculation with vertex correction in W (QSGW^).
    Here we DO clean up the directory before starting the calculation.
    Normally this function is used to write after the self energy k-grid convergence. 
    Collects the output data and checks if the self-consistency cycle has converged.
    If the calculation crashed, this function simply returns 'None'.
    INPUT:
        name:               Name of the material
        nnodes:             Number of nodes to be used for the calculation  
        ncores:             Number of cores to be used for the calculation
        max_iter:           Maximum number of QSGW iteration
    OUTPUT:
        'None' if the calculation crashed...
        OR
        scf_data:           Summary of the self-consistency cycle
                            (list of lists, i.e., [[iteration, rms change in the self energy, band gap], [...]])
        error_flag:         Flag indicating if the self-consistency cycle did not converge
    """
    print("\nRunning a fully self-consistent QSGW^.", flush=True)
    print(f"    Tolerance = 1E-5 Ry", flush=True)
    # prepare the directory for a QSGW^ calculation
    helper.clean_qsgw(name, rst_flag=False, indent=True)
    # optimized the parallelization
    _ = helper.create_pqmap(nnodes, ncores, bse_flag=False) # parallelization for 'lmsig'
    mpi_str = helper.create_pqmap(nnodes, ncores, bse_flag=True) # parallelization for 'bse'
    # run a full self-consistent qsgw^
    misc_str = f"--incount --sym --tol=1e-5 --split-w0 --maxit={max_iter:d}"
    print("    Starting a fully self-consistent QSGW^ calculation.", flush=True)
    execute_command(f"lmgw.sh --bsw {misc_str:s} {mpi_str:s} --bsw ctrl.mat > qsgwbse.log")
    with open("lbsw-b1", "r") as f:
        bsw_str = f.read()
    if "mkw4a: number of threads per group < 1" in bsw_str:
        print("    Parallelization error, try again without pqmap.", flush=True)
        execute_command("rm -f pqmap-* batch-*")
        execute_command("touch meta bz.h5; rm -rf [0-9]*run meta mixm.mat mixsigma")
        print("    Starting a fully self-consistent QSGW^ calculation.", flush=True)
        execute_command(f"lmgw.sh --bsw {misc_str:s} {mpi_str:s} --bsw ctrl.mat > qsgwbse.log")
    # check that the QSGW^ actually finished
    with open("qsgwbse.log", "r") as f:
        last_line = f.readlines()[-1]
    if not re.match(r"lmgw: iter \d+ of \d+ completed in", last_line): 
        print("The QSGW^ has crashed!", flush=True)
        return None
    # retrieve convergence information
    qsqw_rms_info = os.popen("grep RMS qsgwbse.log").read()
    iter_matches = re.findall(r"iter \d+", qsqw_rms_info)
    iter = [int(re.findall(r"\d+", x)[0]) for x in iter_matches]
    rms_matches = re.findall(r"sigma = [-+]?\d*\.?\d+e[-+]?\d+", qsqw_rms_info)
    rms = [float(re.findall(r"[-+]?\d*\.?\d+e[-+]?\d+", x)[0]) for x in rms_matches]
    # get the band gap evolution from the QSGW calculation
    run_dirs = [d for d in os.listdir() if "run" in d and os.path.isdir(d)]
    run_dirs.sort(key=lambda x: int(x.split("run")[0]))
    gaps = []
    for d in run_dirs:
        os.chdir(d)
        gaps += [helper.get_gap("llmf")]
        os.chdir("../")
    # gather everything in one array
    scf_data = np.column_stack([iter, rms, gaps])
    # log message
    print("    Iteration | RMS Sigma (Ry) | Band gap (eV)", flush=True)
    for i in range(len(iter)):
        print(f"    {iter[i]:<9d} | {rms[i]:<14.2E} | {gaps[i]:<2.3f}", flush=True)#
    # convergence check
    if rms[-1] > 1e-5:
        print(
            "Full convergence of the QSGW^ self-consistency cycle was not achieved.",
            flush=True,
        )
        error_flag = 1
    else:
        error_flag = 0
    return scf_data, error_flag
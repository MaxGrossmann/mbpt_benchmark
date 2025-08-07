"""
Functions that just write input files for Yambo.
"""

# external imports
from textwrap import dedent

def write_g0w0(
    bnd_x,
    cutoff_screening,
    bnd_g,
    kpt_bnd_idx,
    ff=100,
    f_name=None,
    gw0_dict={"flag": False, "prev_step": "g0w0"},
    parallel_strategy={"flag": "auto", "ncpu": 1},
    ff_mode={"flag": False},
):
    """
    Creates and adjusts the input file for a Yambo G0W0 calculation in the current directory.
    (THIS INPUT FILE IS BASED ON https://www.nature.com/articles/s41524-023-01027-2)
    The function assumes that the save directory is one directory up.
    INPUT:
        bnd_x:              Number of bands included in the screening
        cutoff_screening:   Number G-vectors in the screening, i.e. the energy cutoff (Ry)
        bnd_g:              Number of bands included in the greens function
        kpt_bnd_idx:        Range kpt1:kpt:2 & bnd1:bnd2 for which the qp energies are calculated (array of length 4)
        f_name:             Custom name for the input file
        gw0_dict:           Dictionary used for self-consistent evGW0 calculations
        parallel_strategy:  Strategy for parallelization of the self energy for GW calculations ("auto" or "yambo")
                            For more information see:
                            https://wiki.yambo-code.eu/wiki/index.php/GW_parallel_strategies
                            "auto" does not change to the input file, only "yambo" does.
    OUTPUT:
        f_name:             Name of the written input file
    """

    # input file name
    if f_name is None:
        f_name = f"g0w0_bndx_{bnd_x:d}_sc_{cutoff_screening:d}_bndg_{bnd_g:d}"

    # create the input file string
    gw_str = dedent(
        f"""\
    #
    # High-Throughput G0W0 by (MG)^2.  
    # YAMBO > 5.0 compatible
    # http://www.yambo-code.org
    #
    rim_cut
    dipoles
    gw0
    HF_and_locXC
    ppa
    em1d
    NLCC
    Chimod = 'hartree'
    % BndsRnXp
    1 | {bnd_x:d} |   
    %
    % GbndRnge
    1 | {bnd_g:d} |   
    %
    % LongDrXp
    1.0 | 1.0 | 1.0 |   
    %
    NGsBlkXp = {cutoff_screening:d} Ry
    % QPkrange
    {kpt_bnd_idx[0]:d} | {kpt_bnd_idx[1]:d} | {kpt_bnd_idx[2]:d} | {kpt_bnd_idx[3]:d} |   
    %
    RandGvec = 100 RL
    RandQpts = 5000024 
    DysSolver = 'n'
    GTermKind = 'BG'
    NLogCPUs = 1
    """
    )
    # full frequency mode
    if ff_mode["flag"] == True:
        gw_str = gw_str.replace("ppa\n", "")
        gw_str = gw_str.replace("Xp", "Xd")
        gw_str += dedent(
            f"""\
            % EnRngeXd
            0.000 | 4.000 |               Ha
            % DmRngeXd
            0.200000 | 0.200000 |         eV    # [Xd] Damping range
            %
            ETStpsXd= {ff:d}                   # [Xd] Total Energy steps
            """
        )

    # self energy parallelization strategy
    if parallel_strategy["flag"] == "yambo":
        ncpu = parallel_strategy["ncpu"]
        gw_str += dedent(
            f"""\
        SE_CPU= '1 {ncpu:d} 1'
        SE_ROLEs= 'q qp b'
        SE_Threads=  0  
        """
        )

    # flag to reuse existing quasiparticle energies in G (for GW0 calculations)
    if gw0_dict["flag"]:
        gw_str += f'GfnQPdb = "E < ./{gw0_dict["prev_step"]}/ndb.QP"'

    # write the input file
    with open(f"{f_name}.in", "w") as f:
        f.write(gw_str)

    return f_name

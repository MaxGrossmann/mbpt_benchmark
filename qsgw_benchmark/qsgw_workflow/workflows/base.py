"""
Common workflow steps are defined by the BaseWorkflow class.
The 'run()' method must be implemented in a subclass, i.e., the actual workflow.
The idea is that one can easily create custom Questaal workflows based on the individual steps defined here.
"""

# external imports
import os
import sys
import shutil
import functools
from time import time
from datetime import datetime

# internal imports
import qsgw_workflow.utils.helper as helper
import qsgw_workflow.utils.runner as runner

class BaseWorkflow:
    """
    Class providing basic functionality required for QSGW/QSGW^ and BSE workflows.
    INPUT:
        calc_path:          Absolute path to the calculation directory
        db_path:            Absolute path to the database directory
        struct_path:        Absolute path to the structure directory
        struct_name:        Name to the structure file
        nnodes:             Number of nodes to be used for the calculation
        ncores:             Number of cores to be used for the calculation (NUMBER OF CORES PER NODE!)
        dft_kppa:           Starting k-point density for the DFT k-grid convergence
        qsgw_kppa:          Starting k-point density for the QSGW k-grid convergence
        dft_tol:            Tolerance for the DFT k-grid convergence (Ry)
        eps_tol:            Tolerance for the dielectric tensor k-grid convergence (SC)
        qsgw_tol:           Tolerance for the QSGW self energy k-grid convergence (Ry)
    """
    # the default convergence parameters are reasonable
    def __init__(
        self, 
        calc_path, 
        db_path, 
        struct_path, 
        struct_name, 
        nnodes,
        ncores,
        dft_kppa=1000, 
        qsgw_kppa=300, 
        dft_tol=1e-4, # Ry
        eps_tol=0.95, # SC
        qsgw_tol=0.00184, # Ry (~25 meV)
    ):
        # initialize parameters
        self.calc_path = calc_path
        self.db_path = db_path
        self.struct_path = struct_path
        self.struct_name = struct_name
        self.name = struct_name # artifact from an older version of the code
        self.nnodes = nnodes
        self.ncores = ncores
        self.dft_kppa = dft_kppa
        self.qsgw_kppa = qsgw_kppa
        self.dft_tol = dft_tol
        self.eps_tol = eps_tol
        self.qsgw_tol = qsgw_tol
        # check that the input parameters are sensible
        self._sanity_checks()
        # flag to check that the workflow order is correct
        self.db_and_calc_dir_flag = False
        # for historical reasons, the number of cores used 
        # in all other parts of the code is the total number of cores;
        # implementing jobs that use more than one node was an afterthought
        self.ncores = self.nnodes * self.ncores
        """
        I have tested standard DFT calculations using 'lmf' with multiple nodes, i.e., 
        4 nodes with 2 cores per node, and using all cores does not really slow down the calculation. 
        I also tested what happens when using more cores than k-points, i.e., I tested a 2x2x2 k-grid 
        (3 irreduciable points for silicon) with 16 cores. I noticed that 'lmf' does not really care 
        and still works normally. So we just use all cores (number of nodes times number of cores per node)
        for the DFT calculations. The communication overhead seems negligible on the Noctua 2 in Paderborn 
        where I tested this. The only time the number of nodes is used is when creating the 'pqmap' for QSGW
        and BSE calculations, as these calculations can really benefit from the extra memory available when 
        using multiple nodes.
        
        In general, it is recommended that you start any calculation on just one node first.
        If memory is an issue, then multiple nodes should be used. 
        Most commonly, QSGW^ and BSE calculations run out of memory.
        """

    def _sanity_checks(self):
        """
        Check that the variables entered are reasonable.
        """
        if self.dft_kppa <= self.qsgw_kppa:
            sys.exit("\n'dft_kppa' should be larger than 'qsgw_kppa'!")
        if self.eps_tol >= 1:
            sys.exit("\nThe SC cannot be equal to or greater than 1. Decrease 'eps_tol'!")

    def init_db_and_calc_dir(self):
        """
        Load an existing database entry or initialize a new one.
        Depending on the state of the database entry, the calculation
        directory will be kept or completely reset. This function also changes
        the current working directory to the calculation directory.
        """
        self.db_file = os.path.join(self.db_path, f"{self.struct_name:s}.json")
        if os.path.exists(self.db_file):
            print(f"\nDatabase entry for {self.struct_name:s} found, reusing convergence parameters.", flush=True)
            try:
                db_entry = helper.load_db_entry(self.db_file)
                self.param_dict = db_entry.parameters
                self.data_dict = db_entry.data
            except Exception as e:
                print(f"Error loading database entry: {e}\nStarting from scratch.", flush=True)
                os.remove(self.db_file)
                self.param_dict = {
                    "dft_kppa_init": self.dft_kppa,
                    "qsgw_kppa_init": self.qsgw_kppa,
                    "dft_tol": self.dft_tol,
                    "eps_tol": self.eps_tol,
                    "qsgw_tol": self.qsgw_tol,
                    "metal_flag_lda": True, # assume everything is a metal
                    "metal_flag_qsgw": True, # assume everything is a metal
                    "qsgw_flag": False,
                    "finish": False,
                }
                self.data_dict = {}
            if (self.param_dict.get("dft_kppa_init") != self.dft_kppa or
                self.param_dict.get("qsgw_kppa_init") != self.qsgw_kppa or
                self.param_dict.get("dft_tol") != self.dft_tol or
                self.param_dict.get("eps_tol") != self.eps_tol or
                self.param_dict.get("qsgw_tol") != self.qsgw_tol):
                print("\nConvergence parameters have changed, removing the database entry.", flush=True)
                os.remove(self.db_file)
                print("Calculating everything from scratch.", flush=True)
                self.param_dict = {
                    "dft_kppa_init": self.dft_kppa,
                    "qsgw_kppa_init": self.qsgw_kppa,
                    "dft_tol": self.dft_tol,
                    "eps_tol": self.eps_tol,
                    "qsgw_tol": self.qsgw_tol,
                    "metal_flag_lda": True, # assume everything is a metal
                    "metal_flag_qsgw": True, # assume everything is a metal
                    "qsgw_flag": False,
                    "finish": False,
                }
                self.data_dict = {}
        else:
            self.param_dict = {
                "dft_kppa_init": self.dft_kppa,
                "qsgw_kppa_init": self.qsgw_kppa,
                "dft_tol": self.dft_tol,
                "eps_tol": self.eps_tol,
                "qsgw_tol": self.qsgw_tol,
                "metal_flag_lda": True, # assume everything is a metal
                "metal_flag_qsgw": True, # assume everything is a metal
                "qsgw_flag": False,
                "finish": False,
            }
            self.data_dict = {}
        # flag to check that the structure, input file and
        # basis set are setup properly before doing anything else
        self.setup_flag = False
        # we always assume that no DFT calculation has been performed in the calculation directory
        self.dft_flag = False # needed to check if, e.g., a band structure can be calculated
        # flag to determine if the self-consistent QSGW starts at the 0th or 1st iteration
        self.fresh_start_flag = True 
        # if a workflow has already been executed, do nothing
        if self.param_dict["finish"] == True:
            sys.exit("\nThe workflow is already finished, so we will stop here!")
        if self.param_dict["qsgw_flag"] == False:
            """
            If no QSGW has been run, start in an empty directory.
            If no convergence parameters change, some steps can be skipped, 
            such as the DFT k-grid convergence or some band structure 
            calculations if their data is already in the database.
            """
            print("\nWe start with a fresh calculation directory.", flush=True)
            files = os.listdir(self.calc_path)
            for f in files:
                if (f == "run.sh") or ("slurm-" in f): # these files are created for or by slurm jobs
                    continue
                f = os.path.join(self.calc_path, f)
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)
        elif self.param_dict["qsgw_flag"] == True:
            """
            Options for restarting the workflow while retaining files from a previous calculation:
                1. The material is a metals:
                    - The workflow may have crashed right after the QSGW or during post-processing steps.
                        -> Band structure, DOS, dielectric function calculation with label QSGW can be 
                           allowed by the 'skip_if_qsgw_done' decorator.
                2. The material is a semiconductor:
                    - The workflow may have crashed right after the QSGW or during post-processing steps.
                        -> Band structure, DOS, dielectric function calculation with label 'QSGW'
                           are allowed by the 'skip_if_qsgw_done' decorator.
                    - Workflow may have crashed during a QSGW^ (most likely due to high memory usage).
                        -> Restart directly at a QSGW^, skip everything else.
            """
            print("\nWe keep all files and continue after the QSGW step.", flush=True)
            # set all variables and flags needed to start a QSGW^,
            # the decorator 'skip_if_qsgw_done' skips all other functions
            self.struct = db_entry.structure
            self.setup_flag = True 
            self.dft_flag = True
            self.setup_flag = True
        # change to the calculation directory
        os.chdir(self.calc_path) # not needed during batch jobs but needed for local calculations
        self.db_and_calc_dir_flag = True
        
    def db_and_calc_dir_check(func):
        """
        Decorator to check if the database entry and 
        the calculation directory are set up correctly.
        """  
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.db_and_calc_dir_flag:
                sys.exit("\nRun 'init_db_and_calc_dir()' before doing anything else!")
            return func(self, *args, **kwargs)
        return wrapper

    def skip_if_qsgw_done(allowed_label=None):
        """
        Decorator to skip a function if the QSGW calculation is already done, i.e., if 'qsgw_flag' is True.
        If 'allowed_label' is provided, the function is only allowed to run if its 'label' argument
        (positional or keyword) matches allowed_label (case-insensitive).
        INPUT:
            allowed_label:      None, str, List of str
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                # check if 'qsgw_flag' is True
                if self.param_dict["qsgw_flag"] == True:
                    if allowed_label is not None:
                        # 'label' must be the first argument (right after 'self')
                        label = args[0]
                        # normalize 'allowed_label' to a list if it's a string.
                        if isinstance(allowed_label, str):
                            allowed = [allowed_label]
                        else:
                            allowed = allowed_label
                        # check if label is among allowed labels (case-insensitive)
                        if label is None or label.lower() not in [a.lower() for a in allowed]:
                            print(
                                f"\nSkipping '{func.__name__:s}()' because 'qsgw_flag' is True " + 
                                f"and label is not in '{allowed_label}'.",
                                flush=True
                            )
                            return True # some functions return 'True' or 'False' depending on whether they finished correctly...
                    else:
                        print(
                            f"\nSkipping '{func.__name__:s}()' because 'qsgw_flag' is set.",
                            flush=True,
                        )
                        return True # some functions return 'True' or 'False' depending on whether they finished correctly...
                # otherwise, run the function normally
                return func(self, *args, **kwargs)
            return wrapper
        return decorator
    
    @skip_if_qsgw_done()
    @db_and_calc_dir_check
    def setup(self):
        """
        Setup the database entry, change to the calculation directory, 
        create a Questaal input file and setup the basis set.
        """
        self.struct = helper.cse2init(self.struct_path, self.struct_name)
        print(f"\nCalculating: {self.struct.composition.reduced_formula:s}", flush=True)
        gmax = helper.init_basis(self.struct)
        self.param_dict["gmax"] = gmax # good to know when comparing with other calculations
        self._save_db_entry()
        self.setup_flag = True
    
    def setup_check(func):
        """
        Decorator to check that the structure, input file
        and basis set are initialized before doing anything else.
        """  
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.setup_flag:
                sys.exit("\nRun 'setup()' before doing anything else!")
            return func(self, *args, **kwargs)
        return wrapper    
        
    @skip_if_qsgw_done()
    @setup_check
    def set_eps_window(self, fpts=5001, emax=5.0):
        """
        Adjust the energy window for the calculation of the dielectric function.
        INPUT:
            fpts:           Number of frequency points
            emax:           Maximum energy (Ha)
        """
        helper.set_eps_window(fpts, emax)
    
    @skip_if_qsgw_done()
    @setup_check
    def dft_kpt_convergence(self):
        """
        Converge the LDA Harris-Foulkes energy with respect to the k-grid.
        """
        if "dft_kpts" in self.param_dict:
            print(f"\nSkipping the DFT k-grid convergence.", flush=True)
        else:
            start_time = time()
            kppa, kpts, conv_data = runner.dft_kpt_conv(
                self.struct, 
                self.dft_kppa, 
                self.ncores, 
                etol=self.dft_tol,
            )
            conv_time = time() - start_time
            self.param_dict["dft_kppa"] = kppa
            self.param_dict["dft_kpts"] = kpts
            self.data_dict["dft_kpt_conv_data"] = conv_data
            self.data_dict["dft_kpt_conv_time"] = conv_time * self.ncores
            self._save_db_entry()
    
    @skip_if_qsgw_done()
    @setup_check
    def dft(self):
        """
        Run a self-consistent DFT calculation (we always use the LDA functional). 
        This function exits if there is a self energy file in the calculation directory,
        because it was just designed to do a DFT using the LDA functional and Questaal 
        automatically reads in a self energy file if it is there.
        """
        if "dft_kpts" not in self.param_dict:
            sys.exit("\nThe workflow needs to run 'dft_kpt_convergence()' before 'dft()'!")
        if os.path.exists("sigm.mat") or os.path.exists("sigm2.mat"):
            sys.exit("\nA self energy file ('sigm.mat'/'sigm2.mat') is in the calculation directory!")
        helper.set_dft_kgrid(self.param_dict["dft_kpts"], indent=False)
        start_time = time()
        runner.run_dft(self.ncores)
        dft_time = time() - start_time
        self.data_dict["dft_time"] = dft_time * self.ncores
        # find which band is the VBM 
        # (this is nice to know and also necessary for the QSGW convergence)
        if "vbm_idx" not in self.param_dict:
            self.param_dict["vbm_idx"] = helper.get_vbm_idx()
        # the band gap is -1 if material is metal
        gap = helper.get_gap("dft.log")
        self.data_dict["gap_lda"] = gap
        if gap == -1:
            self.param_dict["metal_flag_lda"] = True
            print("\nThe calculated material is a metal in the LDA.", flush=True)
        else:
            self.param_dict["metal_flag_lda"] = False
            print(f"\nLDA gap = {gap:2.3f} eV.", flush=True)
        # update the database
        self._save_db_entry()
        # note that a DFT calculation was done
        self.dft_flag = True

    @skip_if_qsgw_done(allowed_label=["qsgw", "qsgwbse"])
    @setup_check
    def bandstructure(self, label, proj_type=None):
        """
        Calculate and parse the band structure. Different projections are possible.
        Due to the limitations of the code, only up to three projections are obtained per calculation.
        Therefore we take the first three atoms with the lowest atomic number Z ('proj_type="atom"').  
        INPUT:
            label:         Identifier of the calculation type, e.g., "lda" or "qsgw"
            proj_type:     None   -> Standard band structure
                           "atom" -> Projection onto individual atoms
        """
        label = label.lower()
        if f"bs_{label:s}" in self.data_dict:
            print(f"\nSkipping the {label.upper():s} band structure calculation.", flush=True)
        else:
            if "dft_kpts" not in self.param_dict:
                sys.exit("\nA workflow must run 'dft_kpt_convergence()' followed by 'dft()' before 'bandstructure()'!")
            if self.dft_flag == False:
                sys.exit("\nA workflow must run 'dft()' before 'bandstructure()'!")
            start_time = time()
            if proj_type is None:
                bs = runner.get_bandstructure(label)
            elif proj_type == "atom":
                bs = runner.get_atom_proj_bandstructure(self.struct, label)
            else:
                sys.exit("\nInvalid band structure calculation type is given!")
            bs_time = time() - start_time
            self.data_dict[f"bs_{label:s}"] = bs
            self.data_dict[f"bs_{label:s}_time"] = bs_time * self.ncores
            self._save_db_entry()
    
    @skip_if_qsgw_done(allowed_label=["qsgw", "qsgwbse"])
    @setup_check
    def pdos(self, label):
        """
        Calculate and parse the total and projected density of states.
        Projections up to "lcut=2", i.e., the d-orbitals, are obtained for all sites. 
        INPUT:
            label:         Identifier of the calculation type, e.g., "lda" or "qsgw"
        """
        label = label.lower()
        if f"dos_{label:s}" in self.data_dict:
            print(f"\nSkipping the {label.upper():s} DOS calculation.", flush=True)
        else:
            if "dft_kpts" not in self.param_dict:
                sys.exit("\nThe workflow must run 'dft_kpt_convergence()' followed by 'dft()' before 'pdos()'!")
            if self.dft_flag == False:
                sys.exit("\nThe workflow must run 'dft()' before 'pdos()'!")
            start_time = time()
            pdos = runner.get_pdos(label)
            dos_time = time() - start_time
            self.data_dict[f"dos_{label:s}"] = pdos
            self.data_dict[f"dos_{label:s}_time"] = dos_time * self.ncores
            self._save_db_entry()
    
    @skip_if_qsgw_done()
    @setup_check
    def ipa_epsilon_kpt_convergence(self):
        """
        Converge the LDA IPA dielectric function with respect to the k-grid.
        This workflow also parses and stores the converged LDA IPA dielectric function.
        """
        if "eps_kpts" in self.param_dict:
            print(f"\nSkipping the LDA IPA dielectric tensor k-grid convergence.", flush=True)
        else:
            start_time = time()
            kppa, kpts, eps, conv_data = runner.eps_kpt_conv(
                self.struct,
                8 * self.param_dict["dft_kppa"], # a k-grid that is twice as dense as the DFT k-grid is a good place to start
                self.ncores,
                sc_tol=self.eps_tol,
            )
            conv_time = time() - start_time
            self.param_dict["eps_kppa"] = kppa
            self.param_dict["eps_kpts"] = kpts
            self.data_dict["eps_conv_data"] = conv_data
            self.data_dict["eps_conv_time"] = conv_time * self.ncores
            self.data_dict["eps_lda"] = eps
            self._save_db_entry()

    @skip_if_qsgw_done(allowed_label=["qsgw", "qsgwbse"])
    @setup_check
    def ipa_epsilon(self, label):
        """
        Calculate and parse the IPA dielectric function.
        INPUT:
            label:          Identifier of the calculation type, e.g., "lda" or "qsgw"
        """
        if "eps_kpts" not in self.param_dict:
            sys.exit("\nThe workflow needs to run 'ipa_epsilon_kpt_convergence()' before 'ipa_epsilon()'!")
        label = label.lower()
        if f"eps_{label:s}" in self.data_dict:
            print(f"\nSkipping the {label.upper():s} IPA dielectric tensor calculation.", flush=True)
        else:
            start_time = time()
            eps = runner.calc_eps(
                self.param_dict["eps_kpts"], 
                self.ncores, 
                label=label.upper(),
            )
            eps_time = time() - start_time
            self.data_dict[f"eps_{label:s}"] = eps
            self.data_dict[f"eps_{label:s}_time"] = eps_time * self.ncores
            self._save_db_entry()

    @skip_if_qsgw_done()
    @setup_check
    def qsgw_kpt_convergence(self):
        """
        Converge the QSGW band gap with respect to the k-grid.
        After the last iteration also does not clean up the calculation directory,
        so we can restart to full self-consistency.
        """
        if "dft_kpts" not in self.param_dict:
            sys.exit("\nThe workflow must run 'dft_kpt_convergence()' before 'qsgw_kpt_convergence()'!")
        # otherwise, the band structure, DOS, and dielectric function 
        # may not be calculated correctly when restarting
        if (
            "qsgw_kpts" in self.param_dict and
            "bs_qpg0w0" in self.data_dict and
            "dos_qpg0w0" in self.data_dict and
            "eps_qpg0w0" in self.data_dict
        ):
            print(f"\nSkipping the QSGW k-grid convergence.", flush=True)
            conv_error_flag = self.param_dict["qsgw_kpt_conv_error_flag"]
        else:
            start_time = time()
            (
                kppa,
                kpts,
                conv_error_flag,
                conv_data,
            ) = runner.qsgw_kpt_conv_semi(
                self.name,
                self.struct,
                self.qsgw_kppa, # initial self energy kppa
                self.param_dict["dft_kppa"], # maximum self energy kppa
                self.nnodes,
                self.ncores,
                etol=self.qsgw_tol,
            )
            conv_time = time() - start_time
            if conv_data[-1][3] == -1: # check the band gap after the last iteration
                self.param_dict["metal_flag_qsgw"] = True # this helps to find problematic materials
                self._save_db_entry()
                sys.exit("\nThe calculated material appears to be a metal in the QSGW!")
            self.param_dict["qsgw_kppa"] = kppa
            self.param_dict["qsgw_kpts"] = kpts
            self.param_dict["qsgw_kpt_conv_error_flag"] = conv_error_flag
            self.data_dict["qsgw_kpt_conv_data"] = conv_data
            self.data_dict["qsgw_kpt_conv_time"] = conv_time * self.ncores
            self._save_db_entry()
            # a subsequent QSGW starts at the 1st iteration, since the 0th is already done
            self.fresh_start_flag = False
        # in general we could just stop here, but we want some results...
        if conv_error_flag:
            # this 'print()' may be replaced by 'sys.exit()'
            print("\nThe k-grid convergence was not achieved when the DFT k-grid was reached.")
        
    @skip_if_qsgw_done(allowed_label=["qsgwbse"])
    @setup_check
    def dft_with_soc_post_gw(self, label):
        """
        Run a self-consistent DFT calculation with spin-orbit coupling and store the band gap.
        This function is optimized to run as a post-processing step after a QSGW or QSGW^ calculation.
        INPUT:
            label:          Identifier of the calculation type, e.g., "qpg0w0" or "qsgwbse" 
                            (This is only used to store the band gap, i.e., it does not work for metals)
        """
        if "dft_kpts" not in self.param_dict:
            sys.exit("\nThe workflow needs to run 'dft_kpt_convergence()' before 'dft()'!")
        if "qsgw_kpts" not in self.param_dict:
            sys.exit("\nThe workflow must run 'qsgw_kpt_convergence()' before 'dft_with_soc_post_gw()'!")
        if f"gap_{label.lower():s}_soc" in self.data_dict:
            print(f"\nSkipping the {label.upper():s}+SOC gap calculation.", flush=True)
        else:
            helper.set_dft_kgrid(self.param_dict["dft_kpts"], indent=False)
            start_time = time()
            runner.run_dft_with_soc_post_gw(self.ncores)
            dft_time = time() - start_time
            self.data_dict["dft_soc_time"] = dft_time * self.ncores
            # the band gap is -1 if material is metal
            gap = helper.get_gap("dft_soc.log")
            self.data_dict[f"gap_{label.lower():s}_soc"] = gap
            if gap == -1:
                print(f"\nThe calculated material is a metal in the {label.upper():s}+SOC.", flush=True)
            else:
                print(f"\n{label.upper():s}+SOC gap = {gap:2.3f} eV.", flush=True)
            # update the database
            self._save_db_entry()
        
    @skip_if_qsgw_done()
    @setup_check      
    def se_kpt_shortcut(self):
        """
        Use the penultimate self energy k-grid for future calculations to save resources.
        IF YOU USE THIS, MAKE SURE THAT THE INITIAL 'qsgw_kppa' IS NOT TOO SMALL.
        """
        if "qsgw_kpt_conv_data" not in self.data_dict:
            sys.exit("\nNo QSGW k-grid convergence data available for the shortcut!")
        print("\nShortcut activated:\nUsing the penultimate self energy k-grid for future QSGW calculations.", flush=True)
        self.fresh_start_flag = True
        conv_data = self.data_dict["qsgw_kpt_conv_data"]
        self.param_dict["qsgw_kppa"] = conv_data[-2][1]
        self.param_dict["qsgw_kpts"] = conv_data[-2][2]
        self._save_db_entry()
        helper.clean_qsgw(self.name, rst_flag=True, indent=False)
       
    @skip_if_qsgw_done()
    @setup_check
    def qsgw(self, max_iter=25):
        """
        Run a self-consistent QSGW calculation.
        INPUT:
            max_iter:       Maximum number of QSGW iteration 
        """
        if "dft_kpts" not in self.param_dict:
            sys.exit("\nThe workflow must run 'dft_kpt_convergence()' before 'qsgw()'!")
        if "qsgw_kpts" not in self.param_dict:
            sys.exit("\nThe workflow must run 'qsgw_kpt_convergence()' before 'qsgw()'!")
        # set the k-grids in the 'ctrl.mat' file
        helper.set_dft_kgrid(self.param_dict["dft_kpts"], indent=False)
        helper.set_gw_kgrid(self.param_dict["qsgw_kpts"], indent=False)
        # run a QSGW
        start_time = time()
        output = runner.run_qsgw(
            self.name, 
            self.nnodes,
            self.ncores,
            max_iter=max_iter,
            fresh_start_flag=self.fresh_start_flag,
        )
        if output is None:
            return False
        scf_data, scf_error_flag = output
        scf_time = time() - start_time
        # gather useful information in the database entry
        self.param_dict["qsgw_max_iter"] = max_iter
        self.param_dict["qsgw_scf_conv_error_flag"] = scf_error_flag
        self.data_dict["qsgw_scf_data"] = scf_data
        self.data_dict["qsgw_time"] = scf_time * self.ncores
        if scf_error_flag:
            print(f"The QSGW self-consistency cycle did not converge after {max_iter:d} iterations!", flush=True)
        gap = scf_data[-1, 2]
        self.data_dict["gap_qsgw"] = gap
        if gap <= 1e-3: # the code returns -1 if the 'lmf' does not find a gap, the 1e-3 is there to catch "strange" materials
            self.param_dict["metal_flag_qsgw"] = True
            print("\nThe calculated material is a metal in the QSGW.", flush=True)
        else:
            self.param_dict["metal_flag_qsgw"] = False
            print(f"\nQSGW gap = {gap:2.3f} eV.", flush=True)
        self.param_dict["qsgw_flag"] = True # remember that QSGW calculation was performed
        self._save_db_entry()
        return True
        
    @setup_check
    def qsgw_with_bse(self, lowest_energy=10, highest_energy=10, max_iter=25):
        """
        Perform a QSGW with vertex corrections in the screened Coulomb interaction W (QSGW^).
        INPUT:
            lowest_energy:      Energy in eV (determines the number of VBs in the BSE Hamiltonian)
            highest_energy:     Energy in eV (determines the number of CBs in the BSE Hamiltonian)
        """
        # sanity checks
        if self.param_dict["qsgw_flag"] == False:
            sys.exit("\nThe workflow must run 'qsgw()' before 'qsgwbse()'!")
        if self.param_dict["metal_flag_qsgw"] == True:
            sys.exit("\nVertex corrections are not supported for metals!")
        if "bs_qsgw" not in self.data_dict:
            sys.exit("\nA QSGW band structure is required to find a suitable transition space!")
        # number of valence bands we want to include in the BSE Hamiltonian
        self.nv = helper.get_bse_vb_manifold(
            self.data_dict["bs_qsgw"], 
            self.param_dict["vbm_idx"], 
            lowest_energy=lowest_energy,
        )
        # number of conduction bands we want to include in the BSE Hamiltonian
        self.nc = helper.get_bse_cb_manifold(
            self.data_dict["bs_qsgw"], 
            self.param_dict["vbm_idx"], 
            highest_energy=highest_energy,
        )
        # this helps with smaller materials where the conduction bands 
        # are spread over a wide energy range, e.g. C, Si, LiF, Ne and so on...
        self.nc = max([self.nc, 8])
        # save the current transition space in the database entry
        self.param_dict["lowest_energy"] = lowest_energy
        self.param_dict["nv"] = self.nv
        self.param_dict["highest_energy"] = highest_energy
        self.param_dict["nc"] = self.nc
        # set the bands in the 'ctrl.mat' files
        helper.set_bse_bands(self.nv, self.nc)
        self._save_db_entry()
        # run a QSGW^
        start_time = time()
        output = runner.run_qsgw_with_bse(
            self.name,
            self.nnodes,
            self.ncores, 
            max_iter=max_iter,
        )
        if output is None:
            return False
        scf_data, scf_error_flag = output
        scf_time = time() - start_time
        # gather useful information in the database entry
        self.param_dict["qsgwbse_max_iter"] = max_iter
        self.param_dict["qsgwbse_scf_conv_error_flag"] = scf_error_flag
        self.data_dict["qsgwbse_scf_data"] = scf_data
        self.data_dict["qsgwbse_time"] = scf_time * self.ncores
        if scf_error_flag:
            print(f"The QSGW^ self-consistency cycle did not converge after {max_iter:d} iterations!", flush=True)
        gap= scf_data[-1, 2]
        print(f"\nQSGW^ gap = {gap:2.3f} eV.", flush=True)
        self.data_dict["gap_qsgwbse"] = gap
        self._save_db_entry()
        return True
                  
    @setup_check  
    def finish(self):
        """
        Update the database and clean up the directory.
        """
        print("\nFinal database update and cleanup.", flush=True)
        self.param_dict["finish"] = True
        self._save_db_entry()
        helper.clean_qsgw(self.name, rst_flag=False, indent=False)
    
    def _save_db_entry(self):
        """
        Save the current workflow parameters and data to the database.
        """
        helper.save_db_entry(self.name, self.db_file, self.struct, self.param_dict, self.data_dict)

    def log(self, message):
        """
        Simple logging utility to include timestamps.
        INPUT:
            message:    String
        """
        print(f"\n{datetime.now()} - {message:s}", flush=True)
    
    def run(self):
        """
        Abstract method that actually executes a workflow. Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the 'run()' method.")
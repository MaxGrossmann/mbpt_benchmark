"""
Class for calculating the band structure, DOS and IPA dielectric function
of semiconductors in the quasiparticle self-consistent GW approximation (QSGW).
After a QSGW calculation this class will also perform a QSGW^ calculation. 
This second step includes vertex corrections in the screened Coulomb interaction W and requires a lot of memory.

For details see: 
https://www.questaal.org/
https://doi.org/10.1103/PhysRevB.108.165104
"""

# external imports
import sys
import numpy as np

# internal imports
import qsgw_workflow.utils.helper as helper
from qsgw_workflow.workflows.base import BaseWorkflow

class SemiconductorWorkflow(BaseWorkflow):
    def run(
        self, 
        shortcut=True,
        fpts=5001, 
        emax=5,
        lowest_energy=10,
        highest_energy=10,
    ):
        """
        INPUT:
            shortcut:           When the self energy k-grid converges, use the previous one to save some resources
            fpts:               Number of frequency points when calculating the dielectric function
            emax:               Maximum energy (Ha) for which the dielectric function is calculated
            lowest_energy:      Energy in eV (determines the number of VBs in the BSE Hamiltonian)
            highest_energy:     Energy in eV (determines the number of CBs in the BSE Hamiltonian)
        """
        # starting log
        self.log(f"Starting QSGW semiconductor workflow using {self.nnodes:d} nodes and {self.ncores // self.nnodes:d} cores per node.")
        
        # initializations
        self.init_db_and_calc_dir()
        self.setup()
        self.set_eps_window(fpts=fpts, emax=emax)
        
        # DFT
        self.dft_kpt_convergence()
        self.dft()
        self.bandstructure("lda", proj_type="atom")
        self.pdos("lda")
        self.ipa_epsilon_kpt_convergence()
        
        # QPG0W0
        self.qsgw_kpt_convergence()
        self.bandstructure("qpg0w0", proj_type="atom")
        self.pdos("qpg0w0")
        self.ipa_epsilon("qpg0w0")
        self.dft_with_soc_post_gw("qpg0w0")
        
        # QSGW self energy k-grid shortcut that saves a lot of resources
        if shortcut:
            self.se_kpt_shortcut()
            qsgw_flag = self.qsgw()
            if qsgw_flag == True:
                self.bandstructure("qsgw", proj_type="atom")
                # sometimes the shortcut can cause artifacts in the band structure, if this happens we undo it
                # (this rarely happens, but it is costly as we have to re-run the QSGW)
                vbm_idx = self.param_dict["vbm_idx"]
                bs = self.data_dict["bs_qsgw"]
                gap = self.data_dict["gap_qsgw"]
                # check that all bands in the band structure are smooth and have no drastic jumps
                # (we have found that only checking the conduction band minimum is sufficient)
                bs_problem_flag = False
                for path in bs["bs_paths"]:
                    band = path["bands"][:, vbm_idx + 1]
                    max_diff = np.max(np.abs(np.diff(band)))
                    if max_diff > 1: # eV
                        bs_problem_flag = True
                        break
                if  gap <= 0 or bs_problem_flag:
                    print("\nThe self energy k-grid seems to be too coarse, the band structure has artifacts/the gap is zero. Undo the shortcut.", flush=True)
                    # reset variables and flags
                    self.data_dict.pop("bs_qsgw", None)
                    self.data_dict.pop("bs_qsgw_time", None)
                    self.param_dict["qsgw_flag"] = False
                    self.fresh_start_flag = True
                    # undo the shortcut
                    conv_data = self.data_dict["qsgw_kpt_conv_data"]
                    self.param_dict["qsgw_kppa"] = conv_data[-1][1]
                    self.param_dict["qsgw_kpts"] = conv_data[-1][2]
                    self._save_db_entry()
                    # restart
                    helper.clean_qsgw(self.name, rst_flag=True, indent=False)
                    self.qsgw()
                    self.bandstructure("qsgw", proj_type="atom")
            else:
                print("\nThe QSGW crashed. The self energy k-grid seems to be too coarse. Undo the shortcut.", flush=True)
                # reset flags
                self.param_dict["qsgw_flag"] = False
                self.fresh_start_flag = True
                # undo the shortcut
                conv_data = self.data_dict["qsgw_kpt_conv_data"]
                self.param_dict["qsgw_kppa"] = conv_data[-1][1]
                self.param_dict["qsgw_kpts"] = conv_data[-1][2]
                self._save_db_entry()
                # restart
                helper.clean_qsgw(self.name, rst_flag=True, indent=False)
                self.qsgw()
                self.bandstructure("qsgw", proj_type="atom")
        else:
            if not self.qsgw():
                sys.exit()
            self.bandstructure("qsgw", proj_type="atom")
                
        # perform the other QSGW post-processing calculations
        self.pdos("qsgw")
        self.ipa_epsilon("qsgw")
        
        # QSGW^
        qsgwbse_flag = self.qsgw_with_bse(lowest_energy=lowest_energy, highest_energy=highest_energy)
        if not qsgwbse_flag: # has probably run out of memory...
            sys.exit()
        self.bandstructure("qsgwbse", proj_type="atom")
        self.pdos("qsgwbse")
        self.ipa_epsilon("qsgwbse")
        self.dft_with_soc_post_gw("qsgwbse")
        
        # end log
        self.finish()
        self.log(f"Finished semiconductor workflow.")
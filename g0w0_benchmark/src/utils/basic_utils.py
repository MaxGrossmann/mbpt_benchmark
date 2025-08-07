"""
Basic utilities which are generally useful in for high-throughput
calculations, but not specific to any single code.
"""

# external imports
import os
import re
import json
import pickle
import textwrap
import subprocess
import numpy as np
import pandas as pd
import pathlib as pl
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.entries.computed_entries import ComputedStructureEntry

# local imports
from src.utils.unit_conversion import ang2au
from src.utils.qe_helper import qe_init_structure

def icsd_load_pickle(data_path_pkl):
    """
    Loads the pickle data for the icsd structures and hold them in memory
    INPUT:
        data_path:      Path of structure data in pickle format
    OUTPUT:
        icsd_data:      Data file with all relevent ICSD structures
    """
    with open(data_path_pkl, "rb") as f:
        icsd_data = pickle.load(f)
    f.close()

    return icsd_data

def get_kpt_grid(structure, kppa):
    """
    Generates the k-point-grid for a structure with given k-point density
    INPUT:
        structure:      The structure for which the k-point-grid should be generated
        kppa:           The k-point density in points per inverse Angstrom^3. If equal to 0, return a gamma-only grid
    OUTPUT:
        kgrid:          A 3x1 list which specifies the number of subdivision in reciprocal space in each dimension
                        (i.e., the normal way to format k-grids). The k-grid is rounded up to the nearest even number
                        to ensure good parallelization
    """
    kpts = Kpoints.automatic_density(structure=structure, kppa=kppa, force_gamma=True)
    if kppa == 0: # for gamma-only calculations
        return [1, 1, 1]
    else:
        kgrid = [i if i % 2 == 0 else i + 1 for i in kpts.kpts[0]]
        return kgrid

def uc_vol_au(structure):
    """
    Calculates the volume of the unit cell in atomic units.
    INPUT:
        structure:      Structure, see pymatgen documentation
    OUTPUT:
        uc_vol:         Volume of the unit cell in atomic units
    """
    uc_vol = structure.volume * ang2au(1) ** 3

    return uc_vol

def init_db(
    base_dir, id, database_dir, structure_load, name, pw_cut_lda
):
    """
    Function to initialize the database entry for a given material.
    INPUT:
        base_dir:           Path to the main directory
        id:                 Database ID
        database_dir:       Name of the database directory
        structure_load:     pymatgen structure object
        name:               Name of the material
        pw_cut_lda:         LDA plane-wave cutoff starting point (Ry)
    OUTPUT:
        db_entry_path:  Path to the database entry
    """

    # check if the database directory exists
    if not os.path.exists(f"{base_dir:s}/{database_dir:s}"):
        os.makedirs(f"{base_dir:s}/{database_dir:s}")

    # database entry path
    db_entry_path = f"{base_dir:s}/{database_dir:s}/{id:s}.json"

    # check if a database file already exists for this material
    if os.path.exists(db_entry_path):
        return db_entry_path

    # initialize structure for a QE calculation
    structure, ibrav = qe_init_structure(structure_load)

    # unit cell volume
    vol = uc_vol_au(structure)

    # create a ComputedStructureEntry
    cse = ComputedStructureEntry(
        structure,
        energy=0.0,  # dummy value
        parameters={
            "id": id,
            "name": name,
            "ibrav": ibrav,
            "vol": vol,
            "lda_pw_cutoff_Ry": pw_cut_lda,
        },
        data={},
        entry_id=id,
    )
    json_dict = cse.as_dict()

    # save the computed structure entry in a JSON file
    with open(db_entry_path, "w+") as f:
        json.dump(json_dict, f, cls=NumpyEncoder)

    return db_entry_path

def start_calc_local(
    base_dir,
    calc_dir,
    database_dir,
    ncores,
    material_id,
    mat,
    workflows,
    pw_cut_lda,
    conda_env="mbpt_benchmark",
):
    """
    Starts a workflow as a local calculation.
    INPUT:
        base_dir:       Path to the main directory
        calc_dir:       Path to calculation directory
        database_dir:   Name of the database directory
        ncores:         Number of cores requested
        material_id:    Database ID
        mat:            ComputedStructureEnrty from pkl of material
        workflows:      List of Workflows wich will be executet
        pw_cut_lda:     LDA plane-wave cutoff starting point (Ry)
        conda_env:      Name of the conda environment to activate
    OUTPUT:
        lsf_name:       Name of the .lsf file
    """

    # setup the calculation directory of the material
    if not os.path.exists(os.path.join(calc_dir, material_id)):
        # create directory
        os.makedirs(os.path.join(calc_dir, material_id))

    # get structure
    structure = mat.structure
    name = mat.reduced_formula

    # initialize the database entry
    db_entry_path = init_db(
        base_dir,
        material_id,
        database_dir,
        structure,
        name,
        pw_cut_lda=pw_cut_lda,
    )

    # write a bash script (adjust the name of the conda environment)
    f = open("run.sh", "w")
    f.write(
        textwrap.dedent(
            f"""\
        #!/bin/bash
        conda activate {conda_env:s}
        pwd
        date
        """
        )
    )
    f.write(f'export PYTHONPATH="{base_dir:s}"\n')
    f.write(
        f'python3 {base_dir:s}/src/do_all_workflow.py {base_dir:s} {ncores:d} {material_id:s} {db_entry_path:s} "{workflows}" {calc_dir:s} 1\n'
    )
    f.write("date")
    f.close()

    # run the bash script
    os.system("bash run.sh")

    # go back to the main directory
    os.chdir(base_dir)

# YOU NEED TO ADJUST THE BATCHJOB PART OF THIS FUNCTION FOR THE SUPERCOMPUTER
# THAT YOU USE, ELSE THIS MAY NOT WORK... THE LOCAL SETUP SHOULD WORK.
def start_calc_lsf(
    base_dir,
    calc_dir,
    job_tag,
    lsf_name,
    database_dir,
    ncores,
    memory,
    material_id,
    mat,
    workflows,
    pw_cut_lda,
    queue,
    conda_env="mbpt_benchmark",
):
    """
    Writes the .lsf file for a given workflow list.
    INPUT:
        base_dir:       Path to the main directory
        calc_dir:       Path to calculation directory
        job_tag:        Tag for the jobs
        lsf_name:       Name of the .lsf file
        database_dir:   Name of the database directory
        ncores:         Number of cores requested
        memory:         Maximum required memory
        material_id:    Database ID
        mat:            ComputedStructureEnrty from the pickle file containing the benchmark materials
        workflows:      List of Workflows wich will be executet
        pw_cut_lda:     LDA plane-wave cutoff starting point (Ry)
        queue:          Queue to submit the job to
        conda_env:      Name of the conda environment to activate
    OUTPUT:
        lsf_name:       Name of the .lsf file
    """

    # setup the calculation directory of the material
    if not os.path.exists(os.path.join(calc_dir, material_id)):
        # create directory
        os.makedirs(os.path.join(calc_dir, material_id))

    # get structure
    structure = mat.structure
    name = mat.reduced_formula

    # go to the calculation directory
    os.chdir(os.path.join(calc_dir, material_id))

    # initialize the database entry
    db_entry_path = init_db(
        base_dir,
        material_id,
        database_dir,
        structure,
        name,
        pw_cut_lda=pw_cut_lda,
    )

    # our cluster uses the IBM Spectrum LSF system
    # the following lines write the BSUB-lsf file
    # CHANGE THIS ACCORDING TO YOUR OWN SETUP
    f = open(lsf_name + ".lsf", "w")
    f.write(
        textwrap.dedent(
            f"""\
        #!/bin/bash
        #BSUB -q {queue:s}
        #BSUB -R "mem > {memory:d} span[hosts=1]"
        #BSUB -oo {material_id:s}_%J.log
        #BSUB -eo {material_id:s}_%J.log
        #BSUB -J {job_tag:s}
        #BSUB -n {ncores:d}
        conda activate {conda_env:s}
        pwd
        date
        """
        )
    )
    f.write(f'export PYTHONPATH="{base_dir:s}"\n')
    f.write(
        f'python3 {base_dir:s}/src/do_all_workflow.py {base_dir:s} {ncores:d} {material_id:s} {db_entry_path:s} "{workflows}" {calc_dir} \n'
    )
    f.write("date")
    f.close()

    # submit the job
    os.system(f"bsub < {lsf_name:s}.lsf")

    # go back to the main directory
    os.chdir(base_dir)

class NumpyEncoder(json.JSONEncoder):
    """
    Special JSON encoder for numpy types
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def icsd_sort_sites(data):
    """
    Sort dataset by ascending number of sites.

    INPUT:
        data:       Dataset
    OUTPUT:
        Dataset sorted by number of sites
    """
    data.sort(key=icsd_get_sites)
    return data

def icsd_get_sites(material):
    """
    Get number of sites for given material

    INPUT:
        material:   CSE of a material
    OUTPUT:
        Number of sites

    """
    return len(material.structure)

def get_job_id():
    """
    Get the job ID from the log file of the calculation.
    This functions assumes that we are in the calculation directory.

    OUTPUT:
    job_id:         LSF job ID
    """

    # get all log file names
    log_files = [f for f in os.listdir() if f.endswith(".log")]

    # find the job with the highest ID
    temp_job_ids = [int(re.findall(r"_(\d+)", f)[0]) for f in log_files]
    job_id = np.max(temp_job_ids, initial=0)
    if job_id == 0:
        raise Exception("No log-file found.")

    return job_id

def get_max_mem(job_id):
    """
    Get the current maximum of memory usage of the current workflow. Need the job id of the calculation running on LSF system.

    INPUT:
        job_id:     Job id of the running calculation
    OUTPUT:
        max_mem:    maximum memory from LSF logfile
    """

    max_mem_str = subprocess.run(
        f'bjobs -l {job_id:d} | grep "MAX MEM"',
        shell=True,
        capture_output=True,
        text=True,
    ).stdout
    max_mem = float(re.findall(r"MAX MEM: (\d+)", max_mem_str)[0])

    return max_mem

def update_if_larger(d, key, new_value):
    """
    Updates a given dictoray if the new value of the given key is larger then the current value of the key.

    INPUT:
        d:          Dictionary
        key:        Key whose value is to be checked
        new_value:  New value whose to be checked
    OUTPUT:
        Dictonaray d with new_value for key if new_values is larger then current value
    """

    if key in d:
        if new_value > d[key]:
            d[key] = new_value
    else:
        d[key] = new_value

def crashed_jobs(path):
    """
    Looks up crashed jobs and returns a pandas DataFrame.

    INPUT:
        path:       Path to the calculation directory
    OUTPUT:
        df_crashed  A Pandas DataFrame containing crashed IDs, exit codes, the maximum required memory, the queue and the exit date
    """
    path = pl.Path(path)

    crashed_ids_list = []
    max_mems_list = []
    exit_code_list = []
    exit_date_list = []
    queue_list = []

    mat_id_list = path.glob("[0-9]*")
    for id in mat_id_list:
        try:
            log_file_list = id.glob("*.log")
        except:
            print(f"No Logfile found for {id}", flush=True)
            continue

        log_file = max(log_file_list, key=lambda x: x.stat().st_ctime)
        with open(log_file, "r") as f:
            log_file_string = f.read()
            errors = re.findall(r".*?Error", log_file_string)
            if (
                len(errors) > 1
                or re.search(r".*?TERM_MEMLIMIT", log_file_string)
                or re.search(r".*?EXIT CODE", log_file_string)
                or re.search(r".*?TERM_OWNER", log_file_string)
            ):
                crashed_ids_list.append(re.findall(r"(\d+)_.*?\.log", log_file.name)[0])
                max_mems_list.append(
                    re.findall(r".*?Max Memory :.*?(\d+\.\d?\d?)", log_file_string)[0]
                )

                exit_date_list.append(
                    re.findall(
                        r".*?Terminated at.*?([a-zA-Z]+ *?\d+ *?\d+:\d+:\d+ *?\d+)",
                        log_file_string,
                    )[0]
                )

                queue_list.append(
                    re.findall(
                        r".*?#BSUB.*?-q.*?([a-zA-Z]+)",
                        log_file_string,
                    )[0]
                )

                if re.search(r".*?EXIT CODE: 9", log_file_string) or re.search(
                    r".*?TERM_MEMLIMIT", log_file_string
                ):
                    exit_code_list.append(9)

                else:
                    # if it is not exit code 9 then set ist to 999
                    exit_code_list.append(999)
        f.close()

    df_crashed = pd.DataFrame(
        {
            "id": crashed_ids_list,
            "crashed_max_mems": max_mems_list,
            "exit_code": exit_code_list,
            "exit_date": exit_date_list,
            "queue": queue_list,
        }
    )

    df_crashed["crashed"] = True

    return df_crashed

"""
THE FUNCTIONs IN THIS SCRIPT ARE SPECIFIC TO THE NOCTUA 2 SUPERCOMPUTER IN PADERBORN.
"""

# external imports
import os
import textwrap

def current_running_tasks(job_name):
    """
    Extracts the number of tasks from a job log string and sums them.
    (The TASK row shows the total, which is the number of cores per node times the number of nodes.)
    INPUT:
        job_name:           Job name (helps to identify a group of jobs)
    OUTPUT:
        total_tasks:        Total number of tasks.
    """
    # log string from 'squeue_pretty' command
    # https://upb-pc2.atlassian.net/wiki/spaces/PC2DOK/pages/1902952/Running+Compute+Jobs
    log_string = os.popen(f"squeue_pretty -n {job_name:s}").read()
    # split the string into lines and find the TASKS column index
    lines = log_string.split("\n")
    tasks_index = None
    for i, line in enumerate(lines):
        if "TASKS" in line:
            # find column index for TASKS
            tasks_index = line.index("TASKS")
            break
    if tasks_index is None:
        raise ValueError("TASKS column not found in the log string.")
    # extract and sum TASKS values from the rows
    total_tasks = 0
    for line in lines[i + 1:]:
        if len(line.strip()) > tasks_index: # ensure the line has enough length
            try:
                task_value = int(line[tasks_index:].split()[0])
                total_tasks += task_value
            except ValueError:
                continue
    return total_tasks

def job_template(
    script_path,
    cargs, 
    nnodes, 
    ncores_per_node,
    memory_per_node=64,
    wall_time=72, 
    job_name="qsgw",
    queue="normal",
    mail_adress="",
    conda_env="mbpt_benchmark",
):
    """
    My SBATCH template string for my jobs on NOCTUA 2 (PC2).
    YOU WILL NEED TO CUSTOMIZE THIS FOR THE SUPERCOMPUTER YOU ARE USING!
    INPUT:
        script_path:        Absolute path to a python script
        cargs:              Command line arguments for the python script
        nnodes:             Number of nodes
        ncores_per_node:    Number of cores per node
        memory_per_node:    Available memory per node (GB)
        wall_time:          Available wall time (h) the job can run for
        job_name:           Job name (helps to identify a group of jobs)
        queue:              Job queue name, see https://upb-pc2.atlassian.net/wiki/spaces/PC2DOK/pages/1902768/Noctua+2+Partitions)
        mail_adress:        Mail address to which an email will be sent when the job finishes
        conda_env:          Name of the conda environment to activate
    OUTPUT:
        job_str:            Job string
    """
    if mail_adress == "":
        mail_type = "NONE"
    else:
        mail_type = "END"
    # YOU NEED NEED TO CHANGE THE PROJECT ID!
    job_str = (
        textwrap.dedent(
        f"""\
        #!/bin/bash 
        #SBATCH -A hpc-prf-cms          # project ID (mandatory!) 
        #SBATCH --job-name={job_name:s}
        #SBATCH -p {queue:s}               # selected queue (normal, largemem, gpu, ...)
        #SBATCH --nodes={nnodes:d}               # number of nodes 
        #SBATCH --ntasks-per-node={ncores_per_node:d} 
        #SBATCH --cpus-per-task=1      
        #SBATCH --mem {int(memory_per_node):d}G
        #SBATCH --time={int(wall_time):d}:00:00         # time (HH:MM:SS)
        #SBATCH --mail-type={mail_type:s}
        #SBATCH --mail-user={mail_adress:s}
        #SBATCH -o slurm-%j.log         # STDOUT
        #SBATCH -e slurm-%j.err         # STDERR
        
        # time and directory logging
        date
        pwd
        
        # this depend on how and where you compiled Questaal
        module purge
        module load toolchain/iimpi/2024a
        module load numlib/imkl-FFTW/2024.2.0-iimpi-2024a
        module load tools/Ninja/1.12.1-GCCcore-13.3.0
        module load devel/CMake/3.29.3-GCCcore-13.3.0
        export PATH="/scratch/hpc-prf-cms/grossmann/lmto-lm-485e2c34ef6d/build:$PATH"
        
        # this depends our python/conda setup
        source /pc2/users/m/magr4985/miniconda3/etc/profile.d/conda.sh
        conda activate {conda_env:s}
        
        # print the nodes we are bound to using
        echo -e "\\nNode and core information"
        srun -l hostname
        
        # start the workflow
        python {script_path:s} {cargs:s}
        
        # time logging
        date
        """
        )
    )
    return job_str
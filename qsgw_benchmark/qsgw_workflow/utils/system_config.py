# external imports
import os
import re
import shlex
import subprocess

# global variable to store the execution mode (the defaut is a 'local' calculation)
execution_mode = "local" 

def set_execution_mode(mode):
    """
    Set the global command execution mode for all 'execute_command' calls.
    INPUT:
        mode:       "local" or "noctua"
    """
    global execution_mode
    if mode not in ["local", "noctua"]:
        raise ValueError("Invalid command execution mode. Choose 'local' or 'noctua'.")
    execution_mode = mode

def get_execution_mode():
    """
    Get the current global execution mode used for all 'execute_command' calls.
    OUTPUT:
        str:        Current execution mode ('local' or 'noctua')
    """
    return execution_mode

def cap_lmf_mpi_tasks(command_str, lmf_max_mpi=16, lmf_max_threads=8):
    """
    Detects 'mpirun -np N lmf ...', caps MPI ranks to 'max_mpi',
    and prefixes with 'env' variables for OpenMP and MKL threading.
    For small to medium systems, the defaults performed well on Noctua 2.
    For larger system or calculations with many k-points the defaults could
    be increased. 
    INPUT:
        command_str:        The command string we want to execute
        lmf_max_mpi:        Maximum number of MPI tasks for an 'lmf' call
        lmf_max_threads:    Maximum number of threads for an 'lmf' call
    OUTPUT:
        command_str:        The command string we want to execute 
                            (with adjusted parallelization if it is an 'lmf' call)
    """
    if "mpirun -np" not in command_str or " lmf" not in command_str:
        return command_str
    parts = command_str.split()
    try:
        idx = parts.index("-np")
        total = int(parts[idx + 1])
    except (ValueError, IndexError):
        return command_str
    ranks = min(lmf_max_mpi, total)
    threads = max(1, min(lmf_max_threads, total // ranks))
    env_prefix = (f"env OMP_NUM_THREADS={threads:d} MKL_NUM_THREADS={threads:d}")
    # replace the original 'np' count
    parts[idx + 1] = str(ranks)
    new_cmd = " ".join(parts)
    return f"{env_prefix:s} {new_cmd:s}"

def execute_command(command_str):
    """
    Executes a command with adjustments based on the global execution mode.
    (The command should redirect the output to a file.)
    INPUT:
        command_str:    The command string we want to execute
    """
    # when running 'lmf' on parallelisation setups with many cores we want to 
    # limit the number of MPI tasks, this is to prevent memory from exploding when
    # using a dense k-point, i.e. when calculating the IPA dielectric function
    command_str = cap_lmf_mpi_tasks(command_str)
    # get the global execution mode
    execution_mode = get_execution_mode()
    # adjustments to run the workflow on the noctua supercomputer
    # https://upb-pc2.atlassian.net/wiki/spaces/PC2DOK/pages/1902952/Running+Compute+Jobs
    if execution_mode == "noctua":
        # get threads
        m = re.search(r"OMP_NUM_THREADS=(\d+)", command_str)
        nt = int(m.group(1)) if m else 1
        if " mpirun -np " in f" {command_str} ":
            m = re.search(r"mpirun\s+-np\s+(\d+)", command_str)
            if m:
                np = int(m.group(1))
                command_str = re.sub(
                    r"\bmpirun\s+-np\s+\d+\b",
                    f"srun -n {np:d} -c {nt:d}",
                    command_str, 
                    count=1,
                )
        elif command_str.startswith(("blm", "lmfa", "lmchk", "lmf", "lmdos", "lmfgwd", "lmgwclear", "kkt", "bse")):
            command_str = f"srun -N 1 -n 1 -c 1 {command_str:s}"  
    # collect the command in a file and run it
    with open("command_history.log", "a+") as f:
        f.write(f"{command_str:s}\n")
    os.system(command_str)
    
def execute_command_timeout(command_str, output_file, max_time=120.0):
    """
    Executes a command with adjustments based on the global execution mode.
    If the maximum execution time is exceeded, we kill the process. This 
    should be used for any short processes that tend to freeze or hang. The
    command should redirect the output to a file. This function is only used
    when generating 'pqmap' or 'pqmap-bse', as these tend to freeze and stall
    the workflow. 
    INPUT:
        command_str:    The command string we want to execute
        output_file:    Name of the output file
        max_time:       Maximum execution time in seconds
    OUTPUT:
        timeout_flag:   1 if timeout, else 0
    """
    # get the global execution mode
    execution_mode = get_execution_mode()
    # adjustments to run the workflow on the noctua supercomputer
    # https://upb-pc2.atlassian.net/wiki/spaces/PC2DOK/pages/1902952/Running+Compute+Jobs
    if execution_mode == "noctua":
        if command_str.startswith("mpirun -np"):
            command_str = command_str.replace("mpirun -np", "srun -n", 1)
        elif command_str.startswith(("blm", "lmfa", "lmchk", "lmf", "lmdos", "lmfgwd", "lmgwclear", "kkt")):
            command_str = f"srun -N 1 -n 1 {command_str:s}"
    # split the command string
    cmd = shlex.split(command_str)
    # collect the command in a file and run it
    with open("command_history.log", "+a") as f:
        f.write(f"{command_str:s}\n")
    # run the command as a subprocess and catch a timeout
    try:
        with open(output_file, "w") as f:
            subprocess.run(cmd, timeout=max_time, stdout=f)
        return 0
    except subprocess.TimeoutExpired:
        print(f"    Command timed out after {max_time:.0f} seconds.")
        return 1
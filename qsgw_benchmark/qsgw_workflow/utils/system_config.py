# external imports
import os
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

def execute_command(command_str):
    """
    Executes a command with adjustments based on the global execution mode.
    (The command should redirect the output to a file.)
    INPUT:
        command_str:    The command string we want to execute
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
    # collect the command in a file and run it
    with open("command_history.log", "+a") as f:
        f.write(f"{command_str:s}\n")
    os.system(command_str)
    
def execute_command_timeout(command_str, output_file, max_time=120.0):
    """
    Executes a command with adjustments based on the global execution mode.
    If the maximum execution time is exceeded, we kill the process.
    This should be used for any short processes that tend to freeze or hang.
    (The command should redirect the output to a file.)
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
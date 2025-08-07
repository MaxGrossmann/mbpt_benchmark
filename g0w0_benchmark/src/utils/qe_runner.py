"""
Functions that can start pw.x calculations
"""

# external imports
import os

def qe_pw_run(
    calc_data,
    input_name,
    ncores,
    errorHandling=True,
    kwargs={},
):
    """
    This function writes and starts a pw.x calculation and does some rudimentary error handling.
    INPUT:
        calc_data:      The calculation that should be run
        input_name:     Which function should be used to write the input (usually a function in qe_write)
        ncores:         How many cores should be used by the calculation
        errorHandling   Check if standard error occurs and handle them (deafault = True)
        kwargs:         Keyword arguments to be passed to the function input_name
    OUTPUT:
        filename:       Name of the input file which was executed
    """

    # set the default diagonalization method
    # (needed for error handling later on, and no other workflow changes this parameter)
    kwargs["diagonalization"] = "david"

    # write the input file for pw.x and start the calculation
    filename = input_name(calc_data, **kwargs)
    os.system(f"mpirun -np {ncores} pw.x -inp {filename}.in > {filename}.out")

    # where we check for various errors and try to fix them
    # we know that these are not all errors that can occur
    # and that some edge cases might not be covered but this
    # is prettx stable for the calculations we are doing

    if errorHandling == True:
        with open(filename + ".out") as f:
            log = f.read()
        if "lone vector" in log:
            print(
                "\nLone vector error detected, increasing pw_cutoff a bit.\n",
                flush=True,
            )
            calc_data.pw_cutoff += 0.1
            filename = input_name(calc_data, **kwargs)
            os.system(f"mpirun -np {ncores} pw.x -inp {filename}.in > {filename}.out")

            # check if the error is still there
            with open(filename + ".out") as f:
                log = f.read()
            if "lone vector" in log:
                open("lone_vector_error.txt", "a").close()
                raise Exception(
                    "\nLone vector error not fixed by increasing pw_cutoff.\n"
                )

        if "convergence NOT achieved" in log:
            print("\nConvergence problem detected, decreasing smearing.\n", flush=True)
            calc_data.degauss /= 3  # reduce smearing from 300 meV to 100 meV
            filename = input_name(calc_data, **kwargs)
            os.system(f"mpirun -np {ncores} pw.x -inp {filename}.in > {filename}.out")

            # check if the error is still there
            with open(filename + ".out") as f:
                log = f.read()
            if "convergence NOT achieved" in log:
                open("conv_iter_error.txt", "a").close()
                raise Exception(
                    "\nConvergence not achieved by reducing the smearing.\n"
                )

        if "Error" in log and not "lone vector" in log:
            print(
                "\nConvergence error, changing the diagonalization method to 'paro'.\n",
                flush=True,
            )
            kwargs["diagonalization"] = "paro"
            filename = input_name(calc_data, **kwargs)
            os.system(f"mpirun -np {ncores} pw.x -inp {filename}.in > {filename}.out")

            # check if the error is still there
            with open(filename + ".out") as f:
                log = f.read()
            if "Error" in log and not "lone vector" in log:
                open("paro_error.txt", "a").close()
                raise Exception("\nDiagonalization not stable even with 'paro'.\n")

        # check if the "eigenvalues not converged" appear
        # but we need to separate two cases
        # 1st case: last iteration of a scf calculation
        # 2nd case: nscf calculation
        with open(filename + ".out") as f:
            # parse the output file
            if "nscf" in filename:
                log = f.read()
            else:
                log = f.read()
                # we only keep the log string after the last iteration
                log = log.split("iteration #")[-1]

        # check if the eigenvalues did not converge
        if "eigenvalues not converged" in log and kwargs["diagonalization"] != "paro":
            kwargs["diagonalization"] = "paro"
            filename = input_name(calc_data, **kwargs)
            os.system(f"mpirun -np {ncores} pw.x -inp {filename}.in > {filename}.out")

            # check if the error is still there
            with open(filename + ".out") as f:
                if "nscf" in filename:
                    log = f.read()
                else:
                    log = f.read()
                    # we only keep the log string after the last iteration
                    log = log.split("iteration #")[-1]
            if "eigenvalues not converged" in log:
                open("eigenvalues_error.txt", "a").close()
                raise Exception(
                    "\nSome eigenvalues are not converged even with 'paro'.\n"
                )

    return filename

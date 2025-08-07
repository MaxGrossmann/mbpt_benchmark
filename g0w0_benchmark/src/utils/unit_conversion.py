"""
This file contains functions for unit conversion.
"""

def ev2ha(val):
    # calculated from CODATA 2006: https://doi.org/10.1103/RevModPhys.80.633
    return val / 27.21138386556469

def ha2ev(val):
    return val * 27.21138386556469

def ang2au(val):
    # calculated from CODATA 2006: https://doi.org/10.1103/RevModPhys.80.633
    return val / 0.5291772083535413

def au2ang(val):
    # calculated from CODATA 2006: https://doi.org/10.1103/RevModPhys.80.633
    return val * 0.5291772083535413

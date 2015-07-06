
"""
Calculate 1D and <3D> atomic abundances for the benchmark stars.
"""

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import cPickle as pickle
import os
from glob import glob
from time import gmtime, strftime

import numpy as np
from astropy.table import Table

import oracle


def _species(element, ion):
    periodictable = """H                                                  He
                       Li Be                               B  C  N  O  F  Ne
                       Na Mg                               Al Si P  S  Cl Ar
                       K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
                       Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
                       Cs Ba Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
                       Fr Ra Lr Rf Db Sg Bh Hs Mt Ds Rg Cn UUt"""
    
    lanthanoids    =  "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb"
    actinoids      =  "Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No"
    
    periodictable = periodictable.replace("Ba ", "Ba " + lanthanoids) \
        .replace("Ra ", "Ra " + actinoids).split()


    atomic_number = periodictable.index(element) + 1
    species = (ion - 1) * 0.1 + atomic_number
    return species


def load_equivalent_widths(filename, metadata=None):
    """
    Load the star name, stellar parameters, and atomic data (including EWs and 
    abundanes) for all benchmark stars.

    :filename:
        The filename (the .dat files provided by Jofre.)

    :param metadata: [optional]
        A dictionary containing additional metadata for this analysis.

    :type metadata:
        dict
    """

    # Parse the star name from the filename.
    star_name = os.path.basename(filename).split("_")[0]

    # Load the stellar parameters.
    with open(filename, "r") as fp:
        # Find the line where the data starts.
        contents = fp.readlines()

    stellar_parameters = {}

    skiprows = 0
    for i, line in enumerate(contents):
        if line.startswith("#"):
            continue

        if "=" in line:
            parameter, value = map(str.strip, line.split("="))
            try:
                stellar_parameters[parameter] = float(value)
            except ValueError:
                print("Could not parse {0} for {1}: '{2}'".format(parameter,
                    star_name, value))
                stellar_parameters[parameter] = np.nan

        if line.startswith("elem "):
            skiprows = i + 1
            break

    else:
        raise WTFError

    # Load the line data.
    line_data = np.loadtxt(filename, skiprows=skiprows, dtype=str)

    atomic_lines = []
    for line in line_data:

        element, ion, wavelength, e_low, log_gf, EW_ULB, log_eps_ULB, EW_BOL, \
            log_eps_BOL = line

        # Change formatting.
        ion = int(ion)
        wavelength, e_low, log_gf, EW_ULB, log_eps_ULB, EW_BOL, log_eps_BOL \
            = map(float, (wavelength, e_low, log_gf, EW_ULB, log_eps_ULB,
                EW_BOL, log_eps_BOL))

        # Put EW_* and log_eps_* as NaNs if not valid.
        if 0 >= EW_BOL:
            EW_BOL = np.nan
        if 0 >= EW_ULB:
            EW_ULB = np.nan
        if 0 >= log_eps_BOL:
            log_eps_BOL = np.nan
        if 0 >= log_eps_ULB:
            log_eps_ULB = np.nan

        # Evaluate species.
        species = _species(element, ion)

        atomic_lines.append({
            "element": element,
            "species": species,
            "ion": ion,
            "wavelength": wavelength,
            "e_low": e_low,
            "log_gf": log_gf,
            "EW_ULB": EW_ULB,
            "log_eps_ULB": log_eps_ULB,
            "EW_BOL": EW_BOL,
            "log_eps_BOL": log_eps_BOL
        })

    # Include default metadata.
    _metadata = {
        "ORIGINAL_FILENAME": filename,
        "PARSED_ORIGINAL_FILENAME_AT_UT": strftime("%Y-%m-%d %H:%M:%S", gmtime()),
        "FORMAT": ("star name", "stellar_parameters", "atomic_lines", "metadata")
    }
    if metadata is not None:
        _metadata.update(metadata)

    return (star_name, stellar_parameters, atomic_lines, _metadata)


def atomic_abundances(star, photosphere_kind, descr_prefix, ew_column, **kwargs):

    interpolator = oracle.photospheres.interpolator(kind=photosphere_kind)

    # Generate a model atmosphere.
    star_name, sp, line_list, metadata = star
    photosphere = interpolator(sp["Teff"], sp["logg"], sp["[Fe/H]"])

    # Link which EWs to use, and format the transitions.
    line_list_copy = [each.copy() for each in line_list]
    for line in line_list_copy:
        line["equivalent_width"] = line[ew_column]
    atomic_transitions = Table(rows=line_list_copy)

    # Calculate the atomic abundances using MOOG.
    abundances = oracle.synthesis.moog.atomic_abundances(
        atomic_transitions, photosphere, microturbulence=sp["vmic"], **kwargs)

    # Use the description_prefix to put all these results in with the existing
    # line data.
    assert len(abundances) == len(atomic_transitions)

    updated_line_list = [each.copy() for each in line_list]
    for atomic_transition, abundance in zip(updated_line_list, abundances):
        atomic_transition["log_eps_{}".format(descr_prefix)] = abundance

    metadata = metadata.copy()
    metadata["CALCULATED_ABUNDANCES_FOR_{}_AT_UT".format(descr_prefix)] \
        = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    metadata["EW_COLUMN_USED_FOR_{}".format(descr_prefix)] = ew_column
    metadata["PHOTOSPHERE_KIND_FOR_{}".format(descr_prefix)] = photosphere_kind
    metadata["KWARGS_FOR_{}".format(descr_prefix)] = kwargs

    return (star_name, sp, updated_line_list, metadata)


if __name__ == "__main__":

    filenames = glob("line-lists/*.dat")

    # Load the stars.
    stars = map(load_equivalent_widths, filenames)

    # Some checks
    for star in stars:
        assert len(star[2]) == len(stars[0][2])


    PHOTOSPHERES = "MARCS"
    DESCRIPTION = "1D_MARCS_ULB_DEFAULT"
    EWS = "EW_ULB"


    analysed_stars = []
    problematic = []
    for star in stars:
        try:
            analysed = atomic_abundances(star, PHOTOSPHERES, DESCRIPTION, EWS,
                damping=1)
        except oracle.synthesis.moog.MOOGException:
            print("Failed on {}".format(star[0]))
            problematic.append(star[0])
            continue
        else:
            # Save the star.
            output_filename = "data/{star}_{descr}.pkl".format(star=star[0],
                descr=DESCRIPTION)
            print("Written to output filename {}".format(output_filename))
            with open(output_filename, "wb") as fp:
                pickle.dump(analysed, fp, -1)

        analysed_stars.append(analysed)



    # Calculate 1D MARCS abundances for each star.

    # Calculate 1D Castelli/Kurucz abundances for each star.

    # Calculate <3D> optical abundances for each star.

    # Calculate <3D> mass density abundances for each star.

    # Calculate <3D> Rosseland abundances for each star.

    # Calculate <3D> geometric height abundances for each star.

    # Save all of the results.
    # (Plots will be produced in some other script).




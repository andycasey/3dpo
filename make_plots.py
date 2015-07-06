#!/usr/bin/env python

""" Produce plots fot 'The impact of using <3D> stellar photospheres' paper. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"


from glob import glob
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

from ew_analysis import load_equivalent_widths

# Initial styling for matplotlib
#matplotlib.rcParams["figure.autolayout"] = True


def load_results(filename):
    with open(filename, "r") as fp:
        results = pickle.load(fp)
    return results

"""
Plots to make:

1) Compare ULB and Bologna EWs
    - as %
    - fraction measured
    - for a giant and for a dwarf
    - for a solar-like star and a cool star

2) Compare ULB and Bologna abundances
    * for just Fe?
    * for other elements
    - compared to REW (strength)
    - compared to \delta_EWs [color by strength?]
    - compared to excitation potential

3) Compare my abundances vs ULB and Bologna
    - # all as per #2
    - any peculiarities?

4) Calculate my abundances in compiled MOOG version.
    * Output photospheres, etc to MOOG format.
    * Parse results and save to pickled format.

5) Assuming mine == MOOG...

6) Investigate abundance discrepancies
    - produce plots, email to ask.

7) Calculate all abundances with different damping treatment

8) Distribution of abundance differences
    - line per line differences in 1D MARCS to 1D CK
    - line per line differences in 1D MARCS to <3D> x,y,z
    - line per line differences in <3D> column mass to <3D> x,y,z
    - 

9) Interpolated photospheric quantities for 1D MARCS/<3D> for the Sun

"""


def plot_ew_difference_percent(filenames, guide_line_color="#666666"):

    
    if isinstance(filenames, (str, unicode)):
        filenames = [filenames]

    stars = map(load_results, filenames)

    measurements = []
    for star in stars:
        name, stellar_parameters, line_list = star[:3]
        measurements.extend(line_list)

    measurements = Table(rows=measurements)

    plt.close("all")
    fig = plt.figure(figsize=(5+.15, 6.00))
    ax2 = plt.subplot2grid((3,2), (0,0), colspan=2)
    ax1 = plt.subplot2grid((3,2), (1,0), colspan=2, rowspan=2)

    ax1.scatter(measurements["EW_ULB"], measurements["EW_BOL"],
        facecolor="k")

    ax2.scatter(measurements["EW_ULB"], 100 * \
        (measurements["EW_BOL"] - measurements["EW_ULB"])/measurements["EW_ULB"],
        facecolor="k")
    
    ax1.set_xlabel("$EW_{\\rm ULB}$ $[{\\rm m}\AA]$")
    ax1.set_ylabel("$EW_{\\rm BOL}$ $[{\\rm m}\AA]$")
    ax2.set_ylabel("$\Delta{EW}_{\\rm BOL-ULB}$ $[\%]$")
    
    lim = ax2.get_xlim()[1]

    # Draw guide lines.
    ax1.plot([0, lim], [0, lim], c=guide_line_color, zorder=-100)
    ax2.axhline(0, c=guide_line_color, zorder=-100)

    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax2.set_xlim(0, lim)

    lim = np.min([100, ax2.get_ylim()[1]])
    ax2.set_ylim(-lim, lim)

    ax2.set_xticklabels([])
    fig = ax1.figure
    fig.tight_layout()

    print("Number of ULB equivalent widths: {0}".format(
        np.isfinite(measurements["EW_ULB"]).sum()))
    print("Number of BOL equivalent widths: {0}".format(
        np.isfinite(measurements["EW_BOL"]).sum()))
    print("Number of EW measurements in common: {0}".format(
        np.isfinite(measurements["EW_ULB"] * measurements["EW_BOL"]).sum()))

    differences = measurements["EW_BOL"] - measurements["EW_ULB"]
    print("Difference (BOL - ULB) mean/median/std.: {0:+.1f} / {1:+.1f} / {2:+.1f} mA"\
        .format(np.nanmean(differences), np.nanmedian(differences),
            np.nanstd(differences)))
    differences *= 100./measurements["EW_ULB"]
    print("Difference (BOL - ULB) mean/median/std.: {0:+.1f} / {1:+.1f} / {2:+.1f} %"\
        .format(np.nanmean(differences), np.nanmedian(differences),
            np.nanstd(differences)))
    
    return fig

"""
2) Compare ULB and Bologna abundances
    * for just Fe?
    * for other elements
    - compared to REW (strength)
    - compared to \delta_EWs [color by strength?]
    - compared to excitation potential
"""

def plot_abundance_differences(filenames, elements=None):

    if isinstance(filenames, (str, unicode)):
        filenames = [filenames]

    stars = map(load_results, filenames)

    measurements = []
    for star in stars:
        name, stellar_parameters, line_list = star[:3]
        measurements.extend(line_list)

    measurements = Table(rows=measurements)

    if elements is not None:
        ok = np.zeros(len(measurements), dtype=bool)
        if isinstance(elements, (str, unicode)):
            elements = [elements]

        for element in elements:
            ok += measurements["element"] == element
        measurements = measurements[ok]

    # Compared to delta difference in ew

    ew_differences = measurements["EW_BOL"] - measurements["EW_ULB"]
    percent_ew_differences = 100 * ew_differences/measurements["EW_ULB"]
    abundance_differences = measurements["log_eps_BOL"] - measurements["log_eps_ULB"]
    bol_rews = np.log(measurements["EW_BOL"]/measurements["wavelength"])

    fig, ax = plt.subplots()
    scat = ax.scatter(bol_rews, abundance_differences, c=(ew_differences),
        vmin=-10, vmax=10)

    cbar = plt.colorbar(scat)
    cbar.set_label("EW_BOL - EW_ULB")
    ax.set_ylim(-1, 1)
    ax.set_xlim(-7.5, -3)
    ax.axhline(0, c="#666666", zorder=-100)

    ax.set_xlabel("REW_ULB")
    ax.set_ylabel("log_eps(BOL) - log_eps(ULB)")

    
    if elements is not None:
        ax.set_title(", ".join(map(str, elements)))

    else:
        ax.set_title("All elements")

    return fig



def plot_my_abundance_differences(filenames, descr, compare="ULB", elements=None,
    mark_elements=True, mark_ionised=True):

    if isinstance(filenames, (str, unicode)):
        filenames = [filenames]

    stars = map(load_results, filenames)

    measurements = []
    for star in stars:
        name, stellar_parameters, line_list = star[:3]
        measurements.extend(line_list)

    measurements = Table(rows=measurements)

    if elements is not None:
        ok = np.zeros(len(measurements), dtype=bool)
        if isinstance(elements, (str, unicode)):
            elements = [elements]

        for element in elements:
            ok += measurements["element"] == element
        measurements = measurements[ok]

    # Compared to delta difference in ew
    abundance_differences = measurements["log_eps_{}".format(descr)]\
        - measurements["log_eps_{}".format(compare)]
    bol_rews = np.log(measurements["EW_{}".format(compare)]/measurements["wavelength"])

    fig, ax = plt.subplots()


    if mark_elements:

        elements = set(measurements["element"])
        for color, element in zip(ax._get_lines.color_cycle, elements):

            _ = measurements["element"] == element

            if mark_ionised:
                ionised = measurements["ion"][_] > 1

                if any(~ionised):
                    ax.scatter(bol_rews[_][~ionised], abundance_differences[_][~ionised],
                        marker="o", facecolor=color, label=element)
                    ax.scatter(bol_rews[_][ionised], abundance_differences[_][ionised],
                        marker="s", facecolor=color)

                else:
                    ax.scatter(bol_rews[_][ionised], abundance_differences[_][ionised],
                        marker="s", facecolor=color, label=element)                

            else:
                ax.scatter(bol_rews[_], abundance_differences[_], marker="o",
                    facecolor="k")

    else:
        if mark_ionised:
            ionised = measurements["ion"] > 1
            scat = ax.scatter(bol_rews[~ionised], abundance_differences[~ionised],
                marker="o", facecolor="k")
            scat = ax.scatter(bol_rews[ionised], abundance_differences[ionised],
                marker="s", facecolor="k")
        else:
            ax.scatter(bol_rews, abundance_differences, marker="o", facecolor="k")
    

    if mark_elements:
        ax.legend()

    #ax.set_ylim(-1, 1)
    #ax.set_xlim(-7.5, -3)
    ax.axhline(0, c="#666666", zorder=-100)

    ax.set_xlabel("REW_{}".format(compare))
    ax.set_ylabel("log_eps(me) - log_eps({})".format(compare))
    
    title = "{0} {1} for ".format(descr, compare)
    title += ", ".join(map(str, elements)) if elements is not None else "all elements"

    ax.set_title(title)
    return fig



"""
ipdb> measurements[112]
<Row 112 of table
 values=(22.5, 22.73, 4.154, 'Fe', 1, 9.429296098632356, 7.41, 7.36, -3.917, 26.0, 5778.45)
 dtype=[('EW_BOL', '<f8'), ('EW_ULB', '<f8'), ('e_low', '<f8'), ('element', 'S2'), ('ion', '<i8'), ('log_eps_1D_MARCS_ULB_DEFAULT', '<f8'), ('log_eps_BOL', '<f8'), ('log_eps_ULB', '<f8'), ('log_gf', '<f8'), ('species', '<f8'), ('wavelength', '<f8')]>

 """




if __name__ == "__main__":

    filenames = glob("data/*3D_MASSDENSITY_ULB_DEFAULT.pkl")

    fig_all_ews = plot_ew_difference_percent(filenames)
    
    # Try for a dwarf vs giant
    fig_18sco = plot_ew_difference_percent("data/18Sco_3D_MASSDENSITY_ULB_DEFAULT.pkl")
    fig_18sco.savefig("ew_difference_18sco.png")

    fig_arcturus = plot_ew_difference_percent("data/Arcturus_3D_MASSDENSITY_ULB_DEFAULT.pkl")
    fig_arcturus.savefig("ew_difference_Arcturus.png")
    
    
    plot_my_abundance_differences("data/18Sco_1D_MARCS_ULB_DEFAULT.pkl",
        "1D_MARCS_ULB_DEFAULT", compare="ULB")
    

    fig_abundances = plot_abundance_differences(filenames, elements="Ti")

    plot_my_abundance_differences("data/18Sco_3D_MASSDENSITY_ULB_DEFAULT.pkl",
        "3D_MASSDENSITY_ULB_DEFAULT", compare="ULB")
    
    plot_my_abundance_differences("data/18Sco_3D_MASSDENSITY_BOL_DEFAULT.pkl",
        "3D_MASSDENSITY_BOL_DEFAULT", compare="BOL")
    

    plot_my_abundance_differences("data/18Sco_1D_MARCS_BOL_DEFAULT.pkl",
        "1D_MARCS_BOL_DEFAULT", compare="BOL")
    

    

    


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of point-source particle release and heatmap creation. 
FVCOM files need to be downloaded first with nc_compressor.py!
"""

import time
import json

t = time.time()  # start the clock
print("Running Scenario: `Surface Plume Transport'.")
import numpy as np
import pandas as pd
from ..oceantracker.main import OceanTracker
import pyproj as proj
from matplotlib import pyplot as plt
import datetime
from ..utils.mapping import (
    plot_heatmap_stats,
)
from oceantracker.post_processing.read_output_files import load_output_files
from oceantracker.post_processing.plotting.plot_tracks import plot_tracks
from matplotlib.transforms import offset_copy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from ..oceantracker.post_processing.plotting import plot_statistics
from ..oceantracker.post_processing.plotting import plot_utilities
from ..oceantracker.post_processing.plotting.plot_utilities import (
    add_credit,
    add_map_scale_bar,
    plot_release_points_and_polygons,
    add_heading,
)
from ..oceantracker.post_processing.plotting.plot_tracks import animate_particles
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as font_manager
import matplotlib
import seaborn as sns
import xarray as xr


def toUTM16(lon, lat, destination_zone=16):
    myproj = proj.Proj(proj="utm", zone=destination_zone, ellips="WGS84")
    return myproj(lon, lat)


def add_grid_JSON(case_info_file):
    with open(case_info_file) as fp:
        listObj = json.load(fp)
    listObj["output_files"]["grid"] = run_name + "_C000_grid.nc"
    listObj["output_files"]["grid_outline"] = (
        run_name + "_C000_grid_outline.json"
    )
    with open(
        case_info_file,
        "w",
    ) as fp:
        json.dump(listObj, fp, indent=4)


def get_run_output_dir(case_info_file):
    with open(case_info_file) as fp:
        listObj = json.load(fp)
    directory = listObj["output_files"]["run_output_dir"]
    return directory


print("Imported packages.")

# %% Scenario Setup

print("Input scenario parameters:")
scenario_title = "Plumes"
date_initial = datetime.datetime(2022, 10, 21, 0, 0)
run_name = scenario_title

# %% Model Run
# make instance of oceantracker
ot = OceanTracker()

ot.settings(
    output_file_base=run_name,  # name used as base for output files
    add_date_to_run_output_dir=True,
    root_output_dir="output",
    # backtracking = True, #choose to run model forward or backwards
    write_tracks=False,  # If True will output very large file giving individual particle locations
    time_step=2**10,  # time step as seconds
    max_run_duration=60 * 60 * 24,  # how long to run the model in seconds
)

# OT elements are organized in classes
ot.add_class(  # read hydrodynamic model output files
    "reader",
    input_dir="./fields",  # folder to search for hindcast files, sub-dirs will, by default, also be searched
    file_mask="lsofs.fields.n00*nc",
    class_name="oceantracker.reader.FVCOM_reader.unstructured_FVCOM",
    cords_in_lat_long=False,
)  # hindcast file mask

ot.add_class(
    "release_groups",
    name="my_release_group",  # provide a name for group
    points=[[-92.00454, 46.71226]],  # [x,y] pairs of release locations
    release_interval=60 * 60 * 24,  # seconds between releasing particles
    release_start_date=date_initial.isoformat(),
    z_range=[-1, 1],  # depth interval to release particles
    pulse_size=2**9,  # number of particles released each release_interval
    max_age=2**22,  # cull particles after a given number of seconds
)

# ot.add_class( #modify sinking/floating
#    "velocity_modifiers",
#    name="upwards_mobility",
#    class_name="oceantracker.velocity_modifiers.terminal_velocity.TerminalVelocity",
#    value=100.0,
# )

ot.add_class(  # generate on-the-fly statistics of particle properties - here, their concentration
    "particle_statistics",
    name="heatmap",  # give the statistic a name
    class_name="oceantracker.particle_statistics.gridded_statistics.GriddedStats2D_timeBased",
    update_interval=60 * 60 * 24,  # update frequency in seconds
    grid_size=[2**8, 2**8],  # spatial resolution
)

# run oceantracker
case_info_file_name = ot.run()
print(case_info_file_name)
# %% Graphing

upper_right = (-84.2, 49.05)
lower_left = (-92.15, 46.3)

lims = [
    toUTM16(lower_left[0], lower_left[1])[0],  # lower left x
    toUTM16(upper_right[0], upper_right[1])[0],  # upper right x
    toUTM16(lower_left[0], lower_left[1])[1],  # lower left y
    toUTM16(upper_right[0], upper_right[1])[1],
]  # upper right y
credit_statement = "Sandborn and Austin 2023 via OceanTracker."
run_name_folder = run_name

plot_heatmap_stats(  # custom plotting function - see mapping.py for details
    case_info_file_name,
    release_group="my_release_group",
    nt=20,
    heatmap_name="heatmap",
    credit="Sandborn and Austin 2024",  # credit_statement,
    plot_size=(6, 3),
    axis_lims=lims,
    cmap="inferno",
    logscale="True",
    colour_bar=True,
    heading="Relative Particle Log-Concentration",
    plot_file_name=case_info_file_name + "_heatmap.png",
)

# %% Return completion text
print(
    "Completed script. Elapsed time: "
    + str(round(time.time() - t))
    + " seconds."
)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapping functions to share among experiments.

Created on Thu Aug  3 11:21:02 2023

@author: sandb425
"""
from matplotlib import pyplot as plt
from matplotlib.transforms import offset_copy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import numpy as np
from oceantracker.post_processing.plotting import plot_utilities
from oceantracker.post_processing.plotting.plot_statistics import (
    _get_stats_data,
)
from oceantracker.post_processing.plotting.plot_utilities import (
    add_credit,
    add_map_scale_bar,
    plot_release_points_and_polygons,
    add_heading,
    plot_coloured_depth,
    text_norm,
)
import cartopy.feature as cfeature
from oceantracker.post_processing.plotting.plot_tracks import animate_particles
from oceantracker.post_processing.read_output_files import load_output_files
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as font_manager
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy import crs
import matplotlib
import seaborn as sns
import pandas as pd
from matplotlib import colors, animation
import pyproj as proj
from oceantracker.util.triangle_utilities_code import (
    convert_face_to_nodal_values,
)

# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from oceantracker.util import time_util
import sys
import datetime

# %% Shared Utilities


def aspect_fixer(
    old_lims, plot_width=6.4, plot_height=4.8, latitude=47
):  # really rough but it works until I can get cartopy on it
    new_height = old_lims[2] + (
        plot_height / plot_width * (old_lims[1] - old_lims[0])
    )
    new_lims = [old_lims[0], old_lims[1], old_lims[2], new_height]
    return new_lims


def toUTM16(lon, lat, destination_zone=16):
    myproj = proj.Proj(proj="utm", zone=destination_zone, ellips="WGS84")
    return myproj(lon, lat)


def convert_utm_to_latlon(df):
    transformer = proj.Transformer.from_crs(26916, 4326)
    lat, lon = transformer.transform(
        df["Easting"].values, df["Northing"].values
    )

    return pd.DataFrame({"Longitude": lon, "Latitude": lat})


def draw_base_map_cartopy(
    grid,
    ax=plt.gca(),
    axis_lims=None,
    back_ground_depth=False,
    show_grid=False,
    back_ground_color_map="Blues",
    title=None,
    text1=None,
    credit=None,
    show_lat_lon_grid=False,
):
    # cartopy outline
    ax.add_feature(cfeature.LAKES, color="blue", alpha=0.05)
    # ax.add_feature(cfeature.BORDERS, ls = ':')
    ax.add_feature(cfeature.STATES, ls="-", edgecolor="grey", alpha=0.3)
    if show_lat_lon_grid:
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

    # if back_ground_depth:
    #    plot_coloured_depth(
    #        grid, ax=ax, color_map=back_ground_color_map, zorder=1
    #    )

    if show_grid:
        ax.triplot(
            grid["x"][:, 0],
            grid["x"][:, 1],
            grid["triangles"],
            color=(0.8, 0.8, 0.8),
            linewidth=0.5,
            zorder=1,
        )

    sel = grid["node_type"] == 3  # open_boundary_nodes
    plt.scatter(
        grid["x"][sel, 0], grid["x"][sel, 1], s=4, marker=".", c="darkgreen"
    )

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", direction="in", right=True, top=True)

    if title is not None:
        ax.set_title(title)
    if text1 is not None:
        text_norm(0.4, 0.1, text1, fontsize=8)
    add_credit(credit)
    add_map_scale_bar(axis_lims, ax=ax)
    return grid


# %% hourly


def plot_tracks_hourly(
    track_data,
    show_grid=False,
    credit=None,
    heading=None,
    figure_size=[7, 7],
    dots_per_inch=200,
    title=None,
    axis_lims=None,
    show_start=False,
    show_hourly=False,
    plot_file_name=None,
    polygon_list_to_plot=None,
    background_map=False,
):
    plt.rcParams["figure.figsize"] = figure_size
    plt.rcParams["figure.dpi"] = 500
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    # Create a background instance.
    map_service = cimgt.OSM()
    crs_proj = ccrs.UTM(zone="16")

    fig = plt.figure()
    # ax = plt.gca()
    ax = fig.add_subplot(1, 1, 1, projection=crs_proj)  # map_service.crs)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    plt.tick_params(
        bottom=True,
        top=True,
        left=True,
        right=True,
        labelbottom=True,
        labeltop=False,
        labelleft=True,
        labelright=False,
        labelrotation=45,
        labelsize="5",
    )

    add_credit(credit)
    add_map_scale_bar(axis_lims, ax=ax)

    # Limit the extent of the map to a small longitude/latitude range.
    ax.set_extent(axis_lims, crs=crs_proj)

    # Add the Stamen data at zoom level 8.
    if background_map:
        ax.add_image(map_service, 11)

    ax.plot(
        track_data["x"][:, :, 0],
        track_data["x"][:, :, 1],
        linewidth=0.1,
        c="k",
        transform=crs_proj,
    )
    if show_hourly:
        hours_of_simulation = int(max(track_data["age"][:, 0]) / 60 / 60 / 3)
        NUM_COLORS = hours_of_simulation
        cm = plt.get_cmap("cividis")

        for i in range(hours_of_simulation):
            minimum_time = int(
                (i + 1) / hours_of_simulation * len(track_data["x"][:, 0, 0])
                - 2
            )
            maximum_time = int(
                (i + 1) / hours_of_simulation * len(track_data["x"][:, 0, 0])
                - 1
            )
            # maximum_time = int((i + 1) / hours_of_simulation * len(track_data['x'][:,0,0]) -1)
            ax.scatter(
                track_data["x"][minimum_time:maximum_time, :, 0],
                track_data["x"][minimum_time:maximum_time, :, 1],
                label=i * 3,
                s=15,
                color=cm(1.0 * i / NUM_COLORS),
                transform=crs_proj,
            )
    if show_start:
        # show all starts, eg random within polygon
        ax.scatter(
            track_data["x0"][:, 0],
            track_data["x0"][:, 1],
            edgecolors=None,
            c="green",
            s=4,
            zorder=8,
            label="Start",
            transform=crs_proj,
        )

    plot_utilities.plot_release_points_and_polygons(
        track_data, ax=ax
    )  # these are nominal starts
    # plot_utilities.draw_polygon_list(polygon_list_to_plot,ax=ax)
    # plot_utilities.show_particleNumbers(track_data['x'].shape[1])
    plot_utilities.add_heading(heading)

    ax.legend(fontsize=6)
    """
    ax.gridlines(
        crs=crs_proj,
        dms=True,
        draw_labels=True,
        linewidth=0.2,
        color="gray",
        alpha=0.4,
        linestyle="--",
        x_inline=False,
    )
    """
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("Northing (UTM 16)", fontsize=5)
    ax.set_xlabel("Easting (UTM 16)", fontsize=5)

    fig.tight_layout()
    if plot_file_name is not None:
        plt.savefig(plot_file_name, dpi=500)
    plt.show()


# %% nps tracks
def plot_tracks_nonpointsource(
    track_data,
    show_grid=False,
    credit=None,
    heading=None,
    figure_size=[7, 7],
    dots_per_inch=200,
    title=None,
    axis_lims=None,
    show_start=False,
    plot_file_name=None,
    polygon_list_to_plot=None,
    background_map=False,
):
    plt.rcParams["figure.figsize"] = figure_size
    plt.rcParams["figure.dpi"] = 500
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    # Create a background instance.
    map_service = cimgt.OSM()
    crs_proj = ccrs.UTM(zone="16")

    fig = plt.figure()
    # ax = plt.gca()
    ax = fig.add_subplot(1, 1, 1, projection=crs_proj)  # map_service.crs)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    plt.tick_params(
        bottom=True,
        top=True,
        left=True,
        right=True,
        labelbottom=True,
        labeltop=False,
        labelleft=True,
        labelright=False,
        labelrotation=45,
        labelsize="5",
    )

    add_credit(credit)
    add_map_scale_bar(axis_lims, ax=ax)

    # Limit the extent of the map to a small longitude/latitude range.
    ax.set_extent(axis_lims, crs=crs_proj)

    # Add the Stamen data at zoom level 8.
    if background_map:
        ax.add_image(map_service, 5)

    ax.plot(
        track_data["x"][:, :, 0],
        track_data["x"][:, :, 1],
        linewidth=0.1,
        c="k",
        transform=crs_proj,
    )
    hours_of_simulation = int(max(track_data["age"][:, 0]) / 60 / 60)
    NUM_COLORS = hours_of_simulation
    cm = plt.get_cmap("cividis")

    # for i in range(hours_of_simulation):
    #   minimum_time = int((i+1) / hours_of_simulation * len(track_data['x'][:,0,0])-2)
    #  maximum_time = int((i+1) / hours_of_simulation * len(track_data['x'][:,0,0])-1)
    # maximum_time = int((i + 1) / hours_of_simulation * len(track_data['x'][:,0,0]) -1)
    # ax.scatter(track_data['x'][minimum_time:maximum_time, :, 0], track_data['x'][minimum_time:maximum_time, :, 1], label = i, s = 15, color =cm(1.*i/NUM_COLORS), transform = crs_proj)
    if show_start:
        # show all starts, eg random within polygon
        ax.scatter(
            track_data["x0"][:, 0],
            track_data["x0"][:, 1],
            edgecolors=None,
            c="green",
            s=4,
            zorder=8,
            label="Start",
            transform=crs_proj,
        )

    plot_utilities.plot_release_points_and_polygons(
        track_data, ax=ax
    )  # these are nominal starts
    # plot_utilities.draw_polygon_list(polygon_list_to_plot,ax=ax)
    # plot_utilities.show_particleNumbers(track_data['x'].shape[1])
    plot_utilities.add_heading(heading)
    ax.legend(title="Hour", fontsize=6)
    ax.gridlines(
        crs=crs_proj,
        dms=True,
        draw_labels=True,
        linewidth=2,
        color="gray",
        alpha=0.4,
        linestyle="--",
        x_inline=False,
    )
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("Northing (UTM 16)", fontsize=5)
    ax.set_xlabel("Easting (UTM 16)", fontsize=5)

    fig.tight_layout()
    if plot_file_name is not None:
        plt.savefig(plot_file_name, dpi=500)
    plt.show()


# %% heatmap tracks
def plot_heatmap_tracks(
    track_data,
    show_grid=False,
    credit=None,
    heading=None,
    figure_size=[7, 7],
    dots_per_inch=200,
    title=None,
    axis_lims=None,
    show_start=False,
    plot_file_name=None,
    polygon_list_to_plot=None,
    background_map=False,
):
    plt.rcParams["figure.figsize"] = figure_size
    plt.rcParams["figure.dpi"] = 500
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    # Create a background instance.
    map_service = cimgt.OSM()
    crs_proj = ccrs.UTM(zone="16")

    fig = plt.figure()
    # ax = plt.gca()
    ax = fig.add_subplot(1, 1, 1, projection=crs_proj)  # map_service.crs)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    plt.tick_params(
        bottom=True,
        top=True,
        left=True,
        right=True,
        labelbottom=True,
        labeltop=False,
        labelleft=True,
        labelright=False,
        labelrotation=45,
        labelsize="5",
    )

    add_credit(credit)
    add_map_scale_bar(axis_lims, ax=ax)

    # Limit the extent of the map to a small longitude/latitude range.
    ax.set_extent(axis_lims, crs=crs_proj)

    # Add the Stamen data at zoom level 8.
    if background_map:
        ax.add_image(map_service, 8)

    maximum_time = len(track_data["x"][:, 0, 0])
    final_positions = pd.DataFrame(
        {
            "x": track_data["x"][maximum_time - 1, :, 0],
            "y": track_data["x"][maximum_time - 1, :, 1],
            "depth": track_data["x"][maximum_time - 1, :, 2],
        }
    )

    sns.scatterplot(data=final_positions, x="x", y="y", s=10, color="k")
    sns.histplot(
        data=final_positions, x="x", y="y", bins=30, pthresh=0.1, cmap="mako"
    )
    sns.kdeplot(
        data=final_positions, x="x", y="y", levels=7, color="w", linewidths=1
    )

    if show_start:
        # show all starts, eg random within polygon
        ax.scatter(
            track_data["x0"][:, 0],
            track_data["x0"][:, 1],
            edgecolors=None,
            c="green",
            s=0.5,
            zorder=8,
            label="Start",
            transform=crs_proj,
        )

    plot_utilities.plot_release_points_and_polygons(
        track_data, ax=ax
    )  # these are nominal starts
    # plot_utilities.draw_polygon_list(polygon_list_to_plot,ax=ax)
    # plot_utilities.show_particleNumbers(track_data['x'].shape[1])
    plot_utilities.add_heading(heading)

    # ax.legend(fontsize = 6)
    ax.gridlines(
        crs=crs_proj,
        dms=True,
        draw_labels=True,
        linewidth=2,
        color="gray",
        alpha=0.4,
        linestyle="--",
        x_inline=False,
    )
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("Northing (UTM 16)", fontsize=5)
    ax.set_xlabel("Easting (UTM 16)", fontsize=5)

    fig.tight_layout()
    if plot_file_name is not None:
        plt.savefig(plot_file_name, dpi=500)
    # plt.show()


# %% From stats file


def plot_heatmap_stats(
    case_info_file,
    release_group,
    heatmap_name="heatmap",
    nt=-1,
    axis_lims=None,
    plot_size=(6.4, 4.8),
    show_grid=False,
    title=None,
    logscale=False,
    colour_bar=True,
    var="count",
    vmin=None,
    plot_polygons=True,
    polylist=None,
    vmax=None,
    credit=None,
    cmap="viridis",
    heading=None,
    dpi=500,
    plot_file_name=None,
    back_ground_depth=False,
    back_ground_color_map=None,
):
    """Plot (and-or save) a heatmap from a saved netCDF heatmap file produced by OT."""
    # todo repace var with data_to_plot=, as in other ploting code
    case_info = case_info_file
    stats_data = load_output_files.load_stats_data(case_info, heatmap_name)
    x, y, z = _get_stats_data(nt, stats_data, var, release_group, logscale)

    # fig = plt.gcf()
    # ax = plt.gca()
    plt.rcParams["figure.dpi"] = dpi
    subplot_kw = {"projection": ccrs.UTM(16)}
    fig, ax = plt.subplots(
        subplot_kw=subplot_kw, figsize=(plot_size[0], plot_size[1])
    )

    ax.set_extent(axis_lims, crs=crs.UTM(16))
    pc = ax.pcolormesh(x, y, z, shading="auto", cmap=cmap, zorder=2)
    if axis_lims is None:
        axis_lims = [
            x[0],
            x[-1],
            y[0],
            y[-1],
        ]  # set axis limits to those of the grid

    draw_base_map_cartopy(
        stats_data["grid"],
        ax=ax,
        axis_lims=axis_lims,
        show_grid=show_grid,
        title=title,
        credit=credit,
        back_ground_depth=back_ground_depth,
        back_ground_color_map=back_ground_color_map,
    )

    pc.set_clim(vmin, vmax)
    if colour_bar:
        plt.colorbar(pc, ax=ax)
    if plot_polygons:
        plot_utilities.plot_release_points_and_polygons(
            stats_data, release_group=release_group, ax=ax
        )
        plot_utilities.draw_polygon_list(
            polylist, ax=plt.gca(), color=[0.2, 0.8, 0.2]
        )

    # plot_utilities.show_particleNumbers(
    #    stats_data["total_num_particles_released"]
    # )
    plot_utilities.add_heading(heading)
    fig.tight_layout()

    plot_utilities.show_output(plot_file_name=plot_file_name)


def plot_heatmap_stats_array(
    case_info_file,
    release_group,
    heatmap_name="heatmap",
    nt=-1,
    axis_lims=None,
    isotimes=[],
    number_cols=1,
    number_rows=1,
    plot_size=(12.8, 9.6),
    show_grid=False,
    title=None,
    logscale=False,
    colour_bar=True,
    var="count",
    vmin=None,
    plot_polygons=True,
    vmax=None,
    credit=None,
    cmap="viridis",
    heading=None,
    dpi=500,
    plot_file_name=None,
    back_ground_depth=False,
    back_ground_color_map=None,
):
    """Plot (and-or save) a heatmap array from a saved netCDF heatmap file produced by OT."""
    # todo repace var with data_to_plot=, as in other ploting code
    if number_cols * number_rows != len(isotimes):
        print("Number of rows * cols must equal number of timestamps to plot!")
        sys.exit(0)

    # turn list of datetimes into timestamp to compare to model output
    timestamper = lambda s: datetime.datetime.fromisoformat(s).timestamp()
    timestamp_list = np.array([timestamper(si) for si in isotimes])

    case_info = case_info_file
    stats_data = load_output_files.load_stats_data(case_info, heatmap_name)

    # find list of nt nearest supplied isotimes
    nt_finder = lambda t: np.absolute(stats_data["time"] - t).argmin()
    nt_list = np.array([nt_finder(ti) for ti in timestamp_list])

    plt.rcParams["figure.dpi"] = dpi
    subplot_kw = {"projection": ccrs.UTM(16)}
    fig, ax = plt.subplots(
        number_rows,
        number_cols,
        subplot_kw=subplot_kw,
        figsize=(plot_size[0], plot_size[1]),
    )
    for i in range(len(isotimes)):  # double check for off-by-one error here
        x, y, z = _get_stats_data(
            nt_list[i], stats_data, var, release_group, logscale
        )
        array_k = i // number_cols  # row?
        array_j = i - array_k * number_cols  # column?
        ax[array_k, array_j].set_extent(axis_lims, crs=crs.UTM(16))
        pc = ax[array_k, array_j].pcolormesh(
            x, y, z, shading="auto", cmap=cmap, zorder=2
        )
        if axis_lims is None:
            axis_lims = [
                x[0],
                x[-1],
                y[0],
                y[-1],
            ]  # set axis limits to those of the grid

        draw_base_map_cartopy(
            stats_data["grid"],
            ax=ax[array_k, array_j],
            axis_lims=axis_lims,
            show_grid=show_grid,
            title=title,
            credit=credit,
            back_ground_depth=back_ground_depth,
            back_ground_color_map=back_ground_color_map,
        )

        pc.set_clim(vmin, vmax)
        if colour_bar:
            plt.colorbar(pc, ax=ax)
        if plot_polygons:
            plot_utilities.plot_release_points_and_polygons(
                stats_data, release_group=release_group, ax=ax
            )

        # plot_utilities.show_particleNumbers(
        #    stats_data["total_num_particles_released"]
        # )
        ax[array_k, array_j].text(
            0.025,
            0.95,
            isotimes[i][0:10],
            fontsize=8,
            transform=ax[array_k, array_j].transAxes,
        )
    # plot_utilities.add_heading(heading)
    fig.tight_layout()

    plot_utilities.show_output(plot_file_name=plot_file_name)


def plot_heatmap_stats_vector(
    run_name,
    run_name_folder,
    release_group,
    nt=-1,
    axis_lims=None,
    isotimes=[],
    number_cols=1,
    plot_size=(12.8, 9.6),
    show_grid=False,
    title=None,
    logscale=False,
    colour_bar=True,
    var="count",
    vmin=None,
    plot_polygons=True,
    vmax=None,
    credit=None,
    cmap="viridis",
    heading=None,
    dpi=500,
    plot_file_name=None,
    back_ground_depth=False,
    back_ground_color_map=None,
):
    """Plot (and-or save) a heatmap array from a saved netCDF heatmap file produced by OT."""
    # todo repace var with data_to_plot=, as in other ploting code
    if number_cols != len(isotimes):
        print("Number of cols must equal number of timestamps to plot!")
        sys.exit(0)

    # turn list of datetimes into timestamp to compare to model output
    timestamper = lambda s: datetime.datetime.fromisoformat(s).timestamp()
    timestamp_list = np.array([timestamper(si) for si in isotimes])

    case_info = (
        "./output/" + run_name_folder + "/" + run_name + "_caseInfo.json"
    )
    stats_data = load_output_files.load_stats_data(case_info, "heatmap")

    # find list of nt nearest supplied isotimes
    nt_finder = lambda t: np.absolute(stats_data["time"] - t).argmin()
    nt_list = np.array([nt_finder(ti) for ti in timestamp_list])

    plt.rcParams["figure.dpi"] = dpi
    subplot_kw = {"projection": ccrs.UTM(16)}
    fig, ax = plt.subplots(
        ncols=number_cols,
        subplot_kw=subplot_kw,
        figsize=(plot_size[0], plot_size[1]),
    )
    for i in range(len(isotimes)):  # double check for off-by-one error here
        x, y, z = _get_stats_data(
            nt_list[i], stats_data, var, release_group, logscale
        )
        ax[i].set_extent(axis_lims, crs=crs.UTM(16))
        pc = ax[i].pcolormesh(x, y, z, shading="auto", cmap=cmap, zorder=2)
        if axis_lims is None:
            axis_lims = [
                x[0],
                x[-1],
                y[0],
                y[-1],
            ]  # set axis limits to those of the grid

        draw_base_map_cartopy(
            stats_data["grid"],
            ax=ax[i],
            axis_lims=axis_lims,
            show_grid=show_grid,
            title=title,
            credit=credit,
            back_ground_depth=back_ground_depth,
            back_ground_color_map=back_ground_color_map,
        )

        pc.set_clim(vmin, vmax)
        if colour_bar:
            plt.colorbar(pc, ax=ax)
        if plot_polygons:
            plot_utilities.plot_release_points_and_polygons(
                stats_data, release_group=release_group, ax=ax
            )

        # plot_utilities.show_particleNumbers(
        #    stats_data["total_num_particles_released"]
        # )
        ax[i].text(
            0.025,
            0.9,
            isotimes[i][:16],
            fontsize=10,
            transform=ax[i].transAxes,
        )
    # plot_utilities.add_heading(heading)
    fig.tight_layout()

    plot_utilities.show_output(plot_file_name=plot_file_name)


# %% Animation
def animate_tracks(
    track_data,
    axis_lims=None,
    colour_using_data=None,
    show_grid=False,
    title=None,
    max_duration=None,
    movie_file=None,
    fps=5,
    dpi=100,
    interval=200,
    size=8,
    polygon_list_to_plot=None,
    min_status=0,
    back_ground_depth=True,
    back_ground_color_map=None,
    credit=None,
    heading=None,
    size_using_data=None,
    part_color_map=None,
    vmin=None,
    vmax=None,
    release_group=None,
    show_dry_cells=False,
    show=True,
):
    print("Beginning animation.")

    def draw_frame(nt):
        if show_dry_cells:
            dry_cell_plot.set_array(dry_cell_data[nt, :])

        # only plot alive particles
        x = track_data["x"][
            nt, :, :2
        ].copy()  # copy so as not to change original data
        sel = (
            track_data["status"][nt, :] < min_status
        )  # get rid of dead particles
        x[sel, :] = np.nan
        sc.set_offsets(x)
        sc.set_array(colour_using_data[nt, :].astype(np.float64))
        sc.set_zorder(5)
        if size_using_data is not None:
            sc.set_sizes(scaled_marker_size[nt, :])
        time_text.set_text(
            time_util.seconds_to_pretty_str(
                track_data["time"][nt], seconds=False
            )
        )
        return sc, time_text, dry_cell_plot

    if max_duration is None:
        num_frames = track_data["time"].shape[0]
    else:
        num_frames = min(
            int(
                track_data["time"].shape[0]
                * max_duration
                / abs(track_data["time"][-1] - track_data["time"][0])
            ),
            track_data["time"].shape[0],
        )

    fig = plt.gcf()

    ax = plt.gca()
    plot_utilities.draw_base_map(
        track_data["grid"],
        ax=ax,
        axis_lims=axis_lims,
        show_grid=show_grid,
        title=title,
        credit=credit,
        back_ground_depth=back_ground_depth,
        back_ground_color_map=back_ground_color_map,
    )

    dry_cell_plot, dry_cell_data = plot_utilities.plot_dry_cells(
        track_data, show_dry_cells
    )

    s0 = size
    nt = num_frames - 1
    if colour_using_data is not None:
        clims = [np.nanmin(colour_using_data), np.nanmax(colour_using_data)]
        cmap = part_color_map
        print("animate_particles: color map limits", vmin, vmax)
        sc = ax.scatter(
            track_data["x"][nt, :, 0],
            track_data["x"][nt, :, 1],
            s=s0,
            c=colour_using_data[nt, :],
            edgecolors=None,
            vmin=clims[0] if vmin is None else vmin,
            vmax=clims[1] if vmax is None else vmax,
            cmap=cmap,
            zorder=5,
        )
    else:
        # colour by status
        colour_using_data = np.full_like(track_data["status"], -127)

        stat_types = track_data["particle_status_flags"]
        status_list = [
            stat_types["outside_open_boundary"],
            stat_types["dead"],
            stat_types["frozen"],
            stat_types["stranded_by_tide"],
            stat_types["on_bottom"],
            stat_types["moving"],
        ]
        for n, val in enumerate(status_list):
            colour_using_data[
                track_data["status"] == val
            ] = n  # replace status with range(status_list)

        colour_using_data = colour_using_data.astype(np.float64)
        status_colour_map = np.asarray(
            [
                [0.6, 0.2, 0.2],
                [0, 0.0, 0.0],
                [0.8, 0, 0.0],
                [0, 0.5, 0.0],
                [0.5, 0.5, 0.5],
                [0, 0, 1.0],
            ]
        )
        cmap = colors.ListedColormap(status_colour_map)

        sc = ax.scatter(
            track_data["x"][nt, :, 0],
            track_data["x"][nt, :, 1],
            c=colour_using_data[nt, :],
            vmin=0,
            vmax=len(status_list),
            s=s0,
            edgecolors=None,
            cmap=cmap,
            zorder=5,
        )

    plot_utilities.plot_release_points_and_polygons(
        track_data, ax=ax, release_group=release_group
    )
    plot_utilities.draw_polygon_list(polygon_list_to_plot, ax=ax)

    if size_using_data is not None:
        # linear sizing on field range
        slims = [np.nanmin(size_using_data), np.nanmax(size_using_data)]
        scaled_marker_size = (
            s0 * (size_using_data - slims[0]) / (slims[1] - slims[0])
        )

    time_text = plt.text(
        0.05,
        0.05,
        time_util.seconds_to_pretty_str(track_data["time"][0], seconds=False),
        transform=ax.transAxes,
    )
    plot_utilities.add_heading(heading)
    plot_utilities.show_particleNumbers(track_data["x"].shape[1])
    fig.tight_layout()

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=num_frames, interval=interval, blit=True
    )
    plot_utilities.animation_output(
        anim, movie_file, fps=fps, dpi=dpi, show=show
    )
    print("Finished animation.")
    return anim


# writergif = matplotlib.animation.PillowWriter(fps = 30)
# anim.save('./output/'+run_name+'/'+run_name+'_movie.gif', writer = writergif)
# this is slow to build!
# HTML(anim.to_html5_video())
# print("Finished animation.")


def animate_stats_heatmap(
    run_name,
    run_name_folder,
    axis_lims=None,
    colour_using_data=None,
    show_grid=False,
    title=None,
    var="count",
    logscale="False",
    max_duration=None,
    movie_file=None,
    colour_bar=True,
    cmap="viridis",
    plot_polygons=False,
    fps=5,
    dpi=150,
    plot_size=(12.8, 9.6),
    interval=200,
    size=8,
    polygon_list_to_plot=None,
    min_status=0,
    back_ground_depth=True,
    back_ground_color_map=None,
    credit=None,
    heading=None,
    size_using_data=None,
    part_color_map=None,
    vmin=None,
    vmax=None,
    release_group=None,
    show_dry_cells=False,
    show=True,
):
    """Plot an animated progression of heatmaps."""
    print("Beginning animation.")

    # pull data from file
    case_info = (
        "./output/" + run_name_folder + "/" + run_name + "_caseInfo.json"
    )
    stats_data = load_output_files.load_stats_data(case_info, "heatmap")
    plt.rcParams["figure.dpi"] = dpi
    subplot_kw = {"projection": ccrs.UTM(16)}
    fig, ax = plt.subplots(
        subplot_kw=subplot_kw,
        figsize=(plot_size[0], plot_size[1]),
    )
    draw_base_map_cartopy(
        stats_data["grid"],
        ax=ax,
        axis_lims=axis_lims,
        show_grid=show_grid,
        title=title,
        credit=credit,
        back_ground_depth=back_ground_depth,
        back_ground_color_map=back_ground_color_map,
    )
    if plot_polygons:
        plot_utilities.plot_release_points_and_polygons(
            stats_data, release_group=release_group, ax=ax
        )

    plot_utilities.draw_polygon_list(polygon_list_to_plot, ax=ax)

    frames = []

    for nt in range(stats_data["time"].shape[0]):
        x, y, z = _get_stats_data(nt, stats_data, var, release_group, logscale)
        ax.set_extent(axis_lims, crs=crs.UTM(16))
        pc = ax.pcolormesh(x, y, z, shading="gouraud", cmap=cmap, zorder=2)
        pc.set_clim(vmin, vmax)

        plot_utilities.show_particleNumbers(
            stats_data["total_num_particles_released"]
        )
        frame_time = ax.text(
            0.025,
            0.95,
            time_util.seconds_to_pretty_str(
                stats_data["time"][nt], seconds=False
            ),
            fontsize=6,
            transform=ax.transAxes,
        )
        frames.append([pc, frame_time])

    # if size_using_data is not None:
    # linear sizing on field range / no idea what this does
    #    slims = [np.nanmin(size_using_data), np.nanmax(size_using_data)]
    #    scaled_marker_size = (
    #        s0 * (size_using_data - slims[0]) / (slims[1] - slims[0])
    #    )

    plot_utilities.add_heading(heading)
    if colour_bar:
        plt.colorbar(
            pc,
            ax=ax,
            label="Relative Particle Concentration, Log Scale",
            pad=0.15,
            shrink=0.7,
        )
    plot_utilities.show_particleNumbers(stats_data["x"].shape[1])
    fig.tight_layout()

    anim = animation.ArtistAnimation(fig, frames, blit=True)
    plot_utilities.animation_output(
        anim, movie_file, fps=fps, dpi=dpi, show=show
    )
    print("Finished animation.")
    return anim

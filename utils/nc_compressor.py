#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:06:38 2023

@author: sandb425
"""

import xarray as xr  # make sure you're using xarray >= 2023.8
import sys

# %%


def thredds_list_constructor(year, month):
    """

    Grabs a list of the LSOFS THREDDS fields.
    Probably fails on leap years before 2020 and after 2024.


    Parameters
    ----------
    year : string
        Format %Y (e.g. "2022").
    month : string
        Format %m (e.g. "09")

    Returns
    -------
    files : list
        A list of all n000-n005 fields in the LSOFS THREDDS folder.

    """
    start_date = int(year + month + "01")
    end_date = int(year + month + "08")  # if it doesn't work, it just passes
    url = (
        "https://www.ncei.noaa.gov/thredds/dodsC/model-lsofs-files/"
        + year
        + "/"
        + month
        + "/nos.lsofs.fields"
    )
    files = [
        f"{url}.{forecast}.{date}.{time}.nc"
        for date in range(start_date, end_date)
        for forecast in ["n000", "n001", "n002", "n003", "n004", "n005"]
        for time in ["t00z", "t06z", "t12z", "t18z"]
    ]
    return files


def thredds_getter(files):
    """

    Gets LSOFS .nc files, while dropping useless variables to minimize filesize.
    Saves files in ./fields


    Parameters
    ----------
    files : list
        A list of all n000-n005 fields in the LSOFS THREDDS folder.

    Returns
    -------
    None.

    """
    suffix = "?nprocs,partition[0:1:174014],x[0:1:90963],y[0:1:90963],lon[0:1:90963],lat[0:1:90963],xc[0:1:174014],yc[0:1:174014],lonc[0:1:174014],latc[0:1:174014],siglay[0:1:19][0:1:90963],siglev[0:1:20][0:1:90963],h[0:1:90963],nv[0:1:2][0:1:174014],iint[0:1:0],time[0:1:0],Itime[0:1:0],Itime2[0:1:0],Times[0:1:0],zeta[0:1:0][0:1:90963],u[0:1:0][0:1:19][0:1:174014],v[0:1:0][0:1:19][0:1:174014],tauc[0:1:0][0:1:174014],ww[0:1:0][0:1:19][0:1:174014],wet_nodes[0:1:0][0:1:90963],wet_cells[0:1:0][0:1:174014],wet_nodes_prev_int[0:1:0][0:1:90963],wet_cells_prev_int[0:1:0][0:1:174014],wet_cells_prev_ext[0:1:0][0:1:174014]"
    for i in files:
        try:
            ds = xr.open_dataset(
                i + suffix,
                drop_variables="Itime2",
                decode_times=True,
                engine="netcdf4",
            )
            # ds.attrs['DODS.strlen'] = 25
            filename = "./fields/demo/" + i[70:109]
            ds.to_netcdf(filename)
        except:
            pass


# thredds_getter(thredds_list_constructor(sys.argv[1], sys.argv[2])) #for use with .sh script
thredds_getter(
    thredds_list_constructor("2024", "04")
)  # grab all fields from April 2024

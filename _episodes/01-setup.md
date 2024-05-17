---
title: Exploratory Data Analysis
teaching: 1
exercises: 0
questions:
- "Understanding data sources"
- "How do I use Python?"
objectives:
- "Brief overview of various data souces"
- "Discuss the benefits and disadvantages of each."
- "Exploratory data analysis and preprocess techniques"
keypoints:
- "Essential libaries for data manipulation and preprocessing"
---

# Visualize Climate data with Python

## Get Metadata

~~~
import xarray as xr

# the line above is necessary for getting 
# your plot embedded within the notebook
%matplotlib inline

dset = xr.open_dataset("data/ERA5_REANALYSIS_precipitation_200306.nc")
print(dset)
~~~
{: .python}

Printing dset returns ERA5_REANALYSIS_precipitation_200306.nc metadata:

~~~
<xarray.Dataset>
Dimensions:    (latitude: 721, longitude: 1440, time: 1)
Coordinates:
  * longitude  (longitude) float32 0.0 0.25 0.5 0.75 ... 359.25 359.5 359.75
  * latitude   (latitude) float32 90.0 89.75 89.5 89.25 ... -89.5 -89.75 -90.0
  * time       (time) datetime64[ns] 2003-06-01
Data variables:
    tp         (time, latitude, longitude) float32 ...
Attributes:
    Conventions:  CF-1.6
    history:      2019-05-31 19:05:13 GMT by grib_to_netcdf-2.10.0: /opt/ecmw...
~~~
{: .output}


We can see that our dset object is an xarray.Dataset, which when printed shows all the metadata associated with our netCDF data file.

In this case, we are interested in the precipitation variable contained within that xarray Dataset:

~~~
print(dset['tp'])
~~~
{: .python}


~~~
xarray.DataArray 'tp' (time: 1, latitude: 721, longitude: 1440)>
[1038240 values with dtype=float32]
Coordinates:
  * longitude  (longitude) float32 0.0 0.25 0.5 0.75 ... 359.25 359.5 359.75
  * latitude   (latitude) float32 90.0 89.75 89.5 89.25 ... -89.5 -89.75 -90.0
  * time       (time) datetime64[ns] 2003-06-01
Attributes:
    units:      m
    long_name:  Total precipitation
~~~
{: .output}


The total precipitation is in units of “metre of water per day”.


## Quick visualization

~~~
dset['tp'].plot()
~~~
{: .python}


We can change the colormap and adjust the maximum (remember the total precipitation is in metre):

~~~
dset['tp'].plot(cmap='jet', vmax=0.02)
~~~
{: .python}


We can see there is a band around the equator and areas especially in Asia and South America with a lot of rain. Let’s add continents and a projection using cartopy:

~~~
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig = plt.figure(figsize=[12,5])

# 111 means 1 row, 1 col and index 1
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))

dset['tp'].plot(ax=ax, vmax=0.02, cmap='jet',
                   transform=ccrs.PlateCarree())
ax.coastlines()

plt.show()
~~~


# Copernicus Climate Data Store (CDS)

## Where to get Climate data?

There are many online services to get climate data, and it is often difficult to know which ones are up-to date and which resources to trust. Also different services provide different Application Programming Interfaces (API), use different terminologies, different file formats etc., which make it difficult for new users to master them all. Therefore in this lesson, we will be focusing on the [Copernicus Climate Change Service (C3S)](https://climate.copernicus.eu/).

## Copernicus Climate Change Service (C3S)

This is a service operated by the [European Centre for Medium-range Weather Forecasts (ECMWF)](https://www.ecmwf.int/) on behalf of the European Union. The [C3S](https://climate.copernicus.eu/) combines observations of the climate system with the latest science to develop authoritative, quality-assured information about the past, current and future states of the climate in Europe and worldwide.

![](fig/C3S_frontpage)

## The Climate Data Store (CDS)

This is a web portal providing a single point of access to a wide range of information. This includes observations (i.e., in-situ measurements, remote sensing data, etc.), historical climate data records, estimates of Essential Climate Variables (ECVs) derived from Earth observations, global and regional climate reanalyses of past observations, seasonal forecasts and climate projections.

### Climate Data Store (CDS) Registration

To be able to use CDS services, you need to [register](https://cds.climate.copernicus.eu/user/login?destination=%2F%23!%2Fhome). Registration to the Climate Data Store (CDS) is free as well as access to climate data.
Before starting, and once registred, login to the Climate Data Store (CDS).

![](fig/CDS_login)



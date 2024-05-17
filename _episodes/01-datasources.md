---
title: Data Sources
teaching: 1
exercises: 0
questions:
- "Understanding data sources"
- "How to get data from online sources"
- "How to retrieve dataset with the Toolbox?"
objectives:
- "Brief overview of various data souces"
- "Discuss the benefits and disadvantages of each."
- "Learn to combine Climate data with your own research topic"
- "Learn how to manipulate netCDF data within the CDS Toolbox"
keypoints:
- "Essential libaries for data online data sources"
- "Data retrieval from the CDS Toolbox"
---

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


## Retrieve Climate data with CDS API

Using CDS web interface is very useful when you need to retrieve small amount of data and you do not need to customize your request. However, it is often very useful to retrieve climate data directly on the computer where you need to run your postprocessing workflow.

In that case, you can use the CDS API (Application Programming Interface) to retrieve Climate data directly in Python from the Climate Data Store.

We will be using `cdsapi` python package.

### Get your API key

- Make sure you login to the [Climate Data Store](https://cds.climate.copernicus.eu/#!/home)

- Click on your username (top right of the main page) to get your API key.
 
![](../fig/get_your_cds_api_key.png)

- Copy the code displayed beside, in the file $HOME/.cdsapirc

~~~
url: https://cds.climate.copernicus.eu/api/v2
key: UID:KEY
~~~
{: .bash}

Where UID is your `uid` and KEY your API key. See [documentation](https://cds.climate.copernicus.eu/api-how-to) to get your API and related information.

### Use CDS API

Once the CDS API client is installed, it can be used to request data from the datasets listed in the CDS catalogue. It is necessary to agree to the Terms of Use of every datasets that you intend to download.

Attached to each dataset download form, the button Show API Request displays the python code to be used. The request can be formatted using the interactive form. The api call must follow the syntax:

~~~
import cdsapi
c = cdsapi.Client()

c.retrieve("dataset-short-name", 
           {... sub-selection request ...}, 
           "target-file")
~~~
{: .python}

For instance to retrieve the same ERA5 dataset e.g. near surface air temperature for June 2003:

![](../fig/CDSAPI_t2m_ERA5.png)

Let’s try it:

~~~
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type':'monthly_averaged_reanalysis',
        'variable':'2m_temperature',
        'year':'2003',
        'month':'06',
        'time':'00:00',
        'format':'netcdf'
    },
    'download.nc')
~~~
{: .python}

### Geographical subset

~~~
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {      
        'area'          : [60, -10, 50, 2], # North, West, South, East. Default: global
        'product_type':'monthly_averaged_reanalysis',
        'variable':'2m_temperature',
        'year':'2003',
        'month':'06',
        'time':'00:00',
        'format':'netcdf'
    },
    'download_small_area.nc')
~~~
{: .python}

### Change horizontal resolution

For instance to get a coarser resolution:
~~~
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {      
        'area'          : [60, -10, 50, 2], # North, West, South, East. Default: global
        'grid'          : [1.0, 1.0], # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
        'product_type':'monthly_averaged_reanalysis',
        'variable':'2m_temperature',
        'year':'2003',
        'month':'06',
        'time':'00:00',
        'format':'netcdf'
    },
    'download_small.nc')
~~~
{: .python}

More information can be found [here](https://confluence.ecmwf.int/display/CKB/C3S+ERA5%3A+Web+API+to+CDS+API).

### To download CMIP 5 Climate data via CDS API

~~~
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'projections-cmip5-monthly-single-levels',
    {
        'variable':'2m_temperature',
        'model':'noresm1_m',
        'experiment':'historical',
        'ensemble_member':'r1i1p1',
        'period':'185001-200512'
    },
    'download_CMIP5.nc')
~~~
{: .python}


> ## Exercise: Download CMIP5 from Climate Data Store with `cdsapi`
> Get near surface air temperature (2m temperature) and precipitation (mean precipitation flux) in one single request and save the result in a file `cmip5_sfc_monthly_1850-200512.zip`.
> What do you get when you unzip this file?
> > ## Solution
> > 
> >  - Download the file 
> >  - Uncompress it
> >  - If you select one variable, one experiment, one model, etc., then you get one file only, and it is a netCDF file (even if it says otherwise!). As soon as you select more than one variable, or more than one experiment, etc., then you get a zip or tgz (depending on the format you chose).
> >
> > ~~~
> > import cdsapi
> > import os
> > import zipfile
> > c = cdsapi.Client()
> > c.retrieve(
> >     'projections-cmip5-monthly-single-levels', 
> >     { 
> >        'variable': ['2m_temperature',
> >       'mean_precipitation_flux'],
> >        'model': 'noresm1_m',
> >         'experiment': 'historical',
> >         'ensemble_member': 'r1i1p1',
> >         'period': '185001-200512',
> >         'format': 'tgz'
> >     },
> >     'cmip5_sfc_monthly_1850-200512.zip'
> > )
> > os.mkdir("./cmip5")
> > with zipfile.ZipFile('cmip5_sfc_monthly_1850-200512.zip', 'r') as zip_ref:
> >     zip_ref.extractall('./cmip5')
> > ~~~
> > {: .python}
> {: .solution}
{: .challenge}

## Climate Data Store Toolbox

Let’s make the same selection as before. Go to ["your requests"](https://cds.climate.copernicus.eu/cdsapp#!/yourrequests?tab=form) tab and select the last product you downloaded.

![](../fig/ERA5_request.png)

Click on “Open request form” and you will be re-directed to the selection page with all your previous choices already pre-selected.

This time, instead of clicking on “Submit Form”, we click on “Show Toolbox request”:

![](../fig/ERA5_show_toolbox)

Copy the content in your clipboard so we can paste it later in the CDS Toolbox.

The CDS Toolbox package is still under active development and the current documentation can be found here.
~~~
mport cdstoolbox as ct  
 data = ct.catalogue.retrieve(  
     'reanalysis-era5-single-levels-monthly-means', 
     {  
         'product_type':'monthly_averaged_reanalysis',  
         'variable':'total_precipitation',  
         'year':'2003', 
         'month':'06',  
         'time':'00:00',    
         'format':'netcdf'  
     }) 
~~~
{: .python}

Then click on “Toolbox” tab to start the CDS toolbox:

![](../fig/ERA5_start_toolbox.png)


- Create a new workspace and name it ERA5_precipitation (make sure you press the enter button to validate your choice otherwise the new workspace will not be created.
- Finally paste your previous selection in the toolbox console:

![](../fig/ERA5_console_toolbox.png)




> ## Is it python syntax?
> If you are a python programmer, you probably have recognized the syntax. Otherwise, it may be a bit difficult to understand! The goal here is not to learn how to use the Python CDS toolbox package as it is currently not open source. For now, we make our selection via the web interface and then copy paste the request.
{: .callout}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:22:50 2022

@author: sambrandt
"""
# MODULES #
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from siphon.catalog import TDSCatalog
from xarray.backends import NetCDF4DataStore
import xarray as xr

# CONSTANTS #
g=9.8
Omega=7.2921*10**-5 # Rotation rate of Earth in rad/s

# INPUTS #
# Edges of Domain (in degrees)
north=55
south=24
east=360-65
west=360-100
# Time of Output (in UTC; code will find the latest GFS run to include it)
year=2022
month=12
day=23
hour=21
# Thermal Wind Barb Spacing (in degrees, must be an integer)
barbstep=2

# FUNCTIONS #
# Function to calculate gradients on a lat/lon grid
def partial(lat,lon,field,wrt):
    gradient=np.zeros(np.shape(field))
    if wrt=='x':
        upper=field[:,2::]
        lower=field[:,0:-2]
        dx=111000*np.cos(lat[:,2::]*(np.pi/180))*(lon[0,1]-lon[0,0])
        grad=(upper-lower)/(2*dx)
        gradient[:,1:-1]=grad
        gradient[:,0]=grad[:,0]
        gradient[:,-1]=grad[:,-1]
    if wrt=='y':
        upper=field[2::,:]
        lower=field[0:-2,:]
        dy=111000*(lat[1,0]-lat[0,0])
        grad=(upper-lower)/(2*dy)
        gradient[1:-1,:]=grad
        gradient[0,:]=grad[0,:]
        gradient[-1,:]=grad[-1,:] 
    return gradient

# DATA DOWNLOAD #
# Define location of the data
best_gfs=TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_onedeg/latest.xml')
best_ds=list(best_gfs.datasets.values())[0]
ncss=best_ds.subset()
# Create a datetime object to specify the output time that you want
valid=datetime(year,month,day,hour)
# Establish a query for the data
query = ncss.query()
# Trim data to location/time of interest
query.lonlat_box(north=north,south=south,east=east,west=west).time(valid)
# Specify that output needs to be in netcdf format
query.accept('netcdf4')
# Specify the variables that you want
query.variables('Geopotential_height_isobaric')
# Retrieve the data using the info from the query
data=ncss.get_data(query)
data=xr.open_dataset(NetCDF4DataStore(data))

# VARIABLE DEFINITIONS #
# Retrieve the geopotential height fields at 600, 500, and 400 mb
gpht=np.array(data['Geopotential_height_isobaric'][0,26:31:2,:,:])
# Define the lat/lon grid with 1 degree spacing
lat=np.arange(south,north+1)
lon=np.arange(west,east+1)
lon,lat=np.meshgrid(lon,lat)
# Define the pressure levels of interest
plev=np.arange(400,700,100)*100
# Tile the array to have the same shape as gpht for element-wise operations
plev=np.repeat(plev[:,np.newaxis],len(lat[:,0]),axis=1)
plev=np.repeat(plev[:,:,np.newaxis],len(lat[0,:]),axis=2)

# CALCULATIONS #
# Planetary vertical vorticity
f=2*Omega*np.sin(lat*(np.pi/180))
# 600 mb geostrophic wind components
ugeo600=-(g/f)*partial(lat,lon,gpht[-1,:,:],'y')
vgeo600=(g/f)*partial(lat,lon,gpht[-1,:,:],'x')
# 500 mb geostrophic wind components
ugeo500=-(g/f)*partial(lat,lon,gpht[1,:,:],'y')
vgeo500=(g/f)*partial(lat,lon,gpht[1,:,:],'x')
# 400 mb geostrophic wind components
ugeo400=-(g/f)*partial(lat,lon,gpht[0,:,:],'y')
vgeo400=(g/f)*partial(lat,lon,gpht[0,:,:],'x')
# 500 mb thermal wind using centered difference
utwn=(ugeo400-ugeo600)/-20000
vtwn=(vgeo400-vgeo600)/20000
# 500 mb absolute vertical vorticity (geostrophic+planetary)
geov500=partial(lat,lon,vgeo500,'x')-partial(lat,lon,ugeo500,'y')+f
# Advection of vorticity by thermal wind
adv=(utwn*partial(lat,lon,geov500,'x')+vtwn*partial(lat,lon,geov500,'y'))

# FIGURE #
# Create cartopy axis
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(),'adjustable': 'box'},dpi=1000)
# Add geographic borders
ax.coastlines(lw=0.25)
ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=0.25)
# Set aspect ratio, establish twin axis for seconday title (further down)
ax2=ax.twinx()
ax.set_box_aspect(len(lat[:,0])/len(lat[0,:]))
# Filled contour plot of thermal wind vorticity advection
pcm=ax.contourf(lon,np.flip(lat,axis=0),adv*10**12,np.arange(-16,17,2),cmap='PuOr')
# Conditional statements to cover for values occasionally peaking the colorbar
if len(adv[adv*10**12<-16])>0:
    ax.scatter(lon[adv*10**12<-16],np.flip(lat)[adv*10**12<-16],transform=ccrs.PlateCarree(),s=1000,zorder=-1,c='#611f00')
if len(adv[adv*10**12>16])>0:
    ax.scatter(lon[adv*10**12>16],np.flip(lat)[adv*10**12>16],transform=ccrs.PlateCarree(),s=1000,zorder=-1,c='#12002e')
# Vorticity contours
ct1=ax.contour(lon,np.flip(lat,axis=0),geov500*10**5,np.arange(20,120,20),colors='black',linewidths=0.25)
ax.clabel(ct1, inline=True, fontsize=4)
# Conditional statement to avoid console warning when there's not strong enough
# anticyclonic vorticity
if len(geov500[geov500<-10]):
    ct2=ax.contour(lon,np.flip(lat,axis=0),geov500*10**5,np.arange(-60,0,20),colors='black',linewidths=0.25,linestyles='dashed')
    ax.clabel(ct2, inline=True, fontsize=4)
# Thermal wind barbs
ax.barbs(lon[::barbstep,::barbstep],np.flip(lat[::barbstep,::barbstep],axis=0),utwn[::barbstep,::barbstep]*10**4,vtwn[::barbstep,::barbstep]*10**4,pivot='tip',length=4,linewidth=0.5,color='black')    
# Create new, dynamically scaled axis for the colorbar
cax = fig.add_axes([ax2.get_position().x1+0.01,ax2.get_position().y0,0.02,ax2.get_position().height])
# Colorbar
cbar = plt.colorbar(pcm,cax=cax)
cbar.ax.set_ylabel('Thermal Wind Vorticity Advection (x$10^{12}$ $m$ $s^{-2}$ $Pa^{-1}$)',fontsize=6)
cbar.ax.tick_params(labelsize=8)  
# Upper title
ax.set_title('500 mb Sutcliffe-Trenberth QG Omega Approximation\nPurple = Forcing for Ascent, Orange = Forcing for Descent\nGFS Initialized '+ncss.metadata.time_span['begin'][0:10]+' '+ncss.metadata.time_span['begin'][11:13]+'z, Valid '+str(valid)[0:13]+'z',fontsize=8)
# Lower title
ax2.get_yaxis().set_visible(False)
ax2.set_title('Absolute Geostrophic Vertical Vorticity (contours, x$10^{5}$ $s^{-1}$)\n600-400 mb Thermal Wind (barbs, x$10^{4}$ $m$ $s^{-1}$ $Pa^{-1}$)\nMade by Sam Brandt (GitHub: SamBrandtMeteo)',y=-0.18,fontsize=8)


Solar-Radiation

from netCDF4 import Dataset
import pylab as plt
import os

root = 'files/data/'

year = 2009

month_list = ['January','February','March','April','May','June',\
              'July','August','September','October','November','December']
              
month_dict = dict(zip(range(1,13),month_list))

for month in range(1,13):
    local_file = root + 'GlobAlbedo.%d%02d.mosaic.5.nc'%(year,month)
    nc = Dataset(local_file,'r')
    band = nc.variables['DHR_SW']
    # access data file
    
out_file = root + 'GlobAlbedo.%d%02d.jpg'%(year,month)

plt.figure()
plt.clf()
plt.title('SW DHR albedo for %9s %d'%(month_dict[month],year))
plt.imshow(band,interpolation='nearest',cmap=plt.get_cmap('Spectral'),vmin=0.0,vmax=1.0)
plt.colorbar()
plt.savefig(out_file)

import os

out_file = 'files/data/GlobAlbedo.%d.SW.1.gif'%year
in_files = out_file.replace('.SW.1.gif','??.gif')

cmd = 'convert -delay 100 -loop 0 %s %s'%(in_files,out_file)
print cmd
os.system(cmd)
# Total SW Albedo in gif format, looped over each month

from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma

root = 'files/data/'
year = 2009

months = xrange(1,13)

data = []

for i,month in enumerate(months):
    local_file = root + 'GlobAlbedo.%d%02d.mosaic.5.nc'%(year,month)
    nc = Dataset(local_file,'r')
    band = np.array(nc.variables['DHR_SW'])
    masked_band = ma.array(band,mask=np.isnan(band))
    data.append(masked_band)
    
data = ma.array(data)

lat = np.array(nc.variables['lat'])

plt.plot(lat)
plt.xlim(0,len(lat))
plt.ylim(-90,90)
plt.title('latitude')
plt.xlabel('column number')
plt.ylabel('latitude')

av_days = 365.25 / 12.
half = av_days/2.
N = np.arange(half,365.25,av_days)
print N

t0 = np.deg2rad (0.98565*(N-2))
t1 = 0.39779*np.cos( np.deg2rad ( 0.98565*(N+10) + 1.914*np.sin ( t0 ) ) )
delta = -np.arcsin ( t1 )

plt.plot(N,np.rad2deg(delta))
plt.xlabel('day of year')
plt.ylabel('solar declination angle (degrees)')
plt.xlim(0,365.25)

h = 0.0

N2 = np.array([[N] * data.shape[1]] * data.shape[2])
print N2.shape

N2 = np.array([[N] * data.shape[1]] * data.shape[2]).T
print N2.shape

lat2 = np.array([np.array([lat] * data.shape[0]).T] * data.shape[2]).T
print lat2.shape

def declination(N):
    t0 = np.deg2rad (0.98565*(N-2))
    t1 = 0.39779*np.cos( np.deg2rad ( 0.98565*(N+10) + 1.914*np.sin ( t0 ) ) )
    delta = -np.arcsin ( t1 )
    return np.rad2deg(delta)
    
def solar_elevation(delta,h,lat):
    lat = np.deg2rad(lat)
    delta = np.deg2rad(delta)
    h = np.deg2rad(h)
    sin_theta = np.cos (h)*np.cos (delta)*np.cos(lat) + np.sin ( delta)*np.sin(lat)
    return np.rad2deg(np.arcsin(sin_theta))
    
N2 = np.array([[N] * data.shape[1]] * data.shape[2]).T
lat2 = np.array([np.array([lat] * data.shape[0]).T] * data.shape[2]).T
h2 = np.zeros_like(N2) + h

delta = declination(N2.copy())
e0 = 1360.
sea = solar_elevation(delta,h2.copy(),lat2.copy())
sin_theta = np.sin(np.deg2rad(sea))
rad = e0*sin_theta

rad[rad < 0] = 0.0

incoming_rad = rad
rad = ma.array(incoming_rad,mask=data.mask)

import numpy as np
np.savez('files/data/solar_rad_data.npz',\
         rad=np.array(rad),data=np.array(data),mask=data.mask)

print rad.sum()
not_valid = np.isnan(rad)
print not_valid
valid = not_valid == False
ndata = rad[valid]
print rad[valid].sum()
#Total Incoming SW Radiation





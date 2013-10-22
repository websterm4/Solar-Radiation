Solar-Radiation

from netCDF4 import Dataset
import pylab as plt
import os

root = 'files/data/'

month_list = ['January','February','March','April','May','June',\
              'July','August','September','October','November','December']
              
month_dict = dict(zip(range(1,13),month_list))

for month in range(1,13):
    local_file = root + 'GlobAlbedo.%d%02d.mosaic.5.nc'%(year,month)
    nc = Dataset(local_file,'r')
    band = nc.variables['DHR_SW']
    
out_file = root + 'GlobAlbedo.%d%02d.jpg'%(year,month)

plt.figure()
plt.clf()
plt.title('SW BHR albedo for %9s %d'%(month_dict[month],year))
plt.imshow(band,interpolation='nearest',cmap=plt.get_cmap('Spectral'),vmin=0.0,vmax=1.0)
plt.colorbar()
plt.savefig(out_file)

import os

out_file = 'files/data/GlobAlbedo.%d.SW.1.gif'%year
in_files = out_file.replace('.SW.1.gif','??.gif')

cmd = 'convert -delay 100 -loop 0 %s %s'%(in_files,out_file)
print cmd
os.system(cmd)

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
    
data = np.array(data)






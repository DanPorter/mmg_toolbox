"""
mmg_toolbox example
Example script to load multiple files from a 2D grid scan
"""

import numpy as np
import matplotlib.pyplot as plt
from mmg_toolbox import Experiment


exp = Experiment('/dls/science/groups/das/ExampleData/i16/2dscans_2026/', instrument='i16')

# Add custom ROI
exp.add_roi('new_roi', cen_i=95, cen_j=200, wid_i=40, wid_j=45, image_name='pil3_100k')

scan_numbers = range(1144247, 1144272)
scans = exp.scans(*scan_numbers)
ttl = exp.generate_scans_title(*scan_numbers)

xx, yy, image = exp.generate_mesh(*scan_numbers, axes='sx', values='sy', signal='signal')

plt.figure()
plt.pcolormesh(xx, yy, image)
plt.axis('image')
cb = plt.colorbar(label='Intensity')
plt.xlabel('sx',fontsize=22)
plt.ylabel('sy',fontsize=22)
plt.title(ttl,fontsize=18)

"""
# perform fits and extract data
sx_values = np.zeros(len(scans))
sy_values = np.zeros(len(scans))
area_values = np.zeros(len(scans))
for n, scan in enumerate(scans):
    s = scan.map.get_image_shape()
    scan.map.add_roi('pilsum', s[0]//2, s[1]//2, s[0], s[1])  # add ROI for whole detector
    result = scan.fit.multi_peak_fit(
        xaxis='axes',  # default scan axes
        yaxis='pilsum_total / Transmission / (rc / 300)', #'signal',  # default scan values
        npeaks=None, # determine the number of peaks automatically
        min_peak_power=None,
        peak_distance_idx=6,
        model='Gaussian',
        background='Slope',
        plot_result=True
    )
    sx_values[n] = scan('mean(sx)')
    sy_values[n] = scan('mean(sy)')
    area_values[n] = result.amplitude  # total area under all peaks
    #area_values[n] = result.fwhm  # width
    #area_values[n] = result.center  # centre


# Determine the repeat length of the scans
delta_x = np.abs(np.diff(sx_values))
ch_idx_x = np.where(delta_x > delta_x.max()*0.9) # find biggest changes
ch_delta_x = np.diff(ch_idx_x)
rep_len_x = np.round(np.mean(ch_delta_x))
delta_y = np.abs(np.diff(sy_values))
ch_idx_y = np.where(delta_y > delta_y.max()*0.9) # find biggest changes
ch_delta_y = np.diff(ch_idx_y)
rep_len_y = np.round(np.mean(ch_delta_y))
print('Scans in sx are repeating every {} iterations'.format(rep_len_x))
print('Scans in sy are repeating every {} iterations'.format(rep_len_y))
rep_len = int(max(rep_len_x, rep_len_y))

# Reshape into square arrays
# If this is problematic, look at scipy.interpolate.griddata
sx_squareA = sx_values[:rep_len*(len(sx_values)//rep_len)].reshape(-1,rep_len)
sy_squareA = sy_values[:rep_len*(len(sy_values)//rep_len)].reshape(-1,rep_len)
area_squareA = area_values[:rep_len*(len(area_values)//rep_len)].reshape(-1,rep_len)

plt.figure(figsize=(12, 10), dpi=60)
plt.pcolormesh(sx_squareA, sy_squareA, area_squareA, shading='auto')
plt.axis('image')
cb = plt.colorbar(label='Amplitude')
plt.xlabel('sx',fontsize=22)
plt.ylabel('sy',fontsize=22)
plt.title(ttl,fontsize=18)
cb.set_label('Integrated Area',fontsize=22)
"""
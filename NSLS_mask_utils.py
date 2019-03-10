#!/python -u
'''
NSLS_mask_utils is developed to create masks for the diffraction imgase

collected from the Eiger detector.

Started 3-9-2019 @ NSLS FMX

Authors: Chufeng Li

E-mail: chufengl@asu.edu

Usage:
	NSLS_mask_utils.py <Eiger_file_name> <mask_name>
	
<Eiger_file_name>	the image file collected from Eiger detector
<mask_name>		the .h5 mask file for the beam stop, default: beam_stop_mask.h5

'''

import sys,os
sys.path.append('/GPFS/CENTRAL/xf17id1/mfuchs/CFL/dev')
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, feature
import scipy
import glob
import h5py

def beam_stop_mask(Eiger_file_name,thld=100,mask_name='beam_stop_mask.h5'):

	import NSLS_FMX_utils as utils
	
	dsum=h5py.File(Eiger_file_name,'r')
	img_arry=dsum['entry/data/data'].value.sum(axis=0)
	all_labels=measure.label(img_arry<thld)
	props=measure.regionprops(all_labels,img_arry)
	area=np.array([r.area for r in props]).reshape(-1,)
	max_intensity=np.array([r.max_intensity for r in props]).reshape(-1,)
	label=np.array([r.label for r in props]).reshape(-1,)
	min_pix=50
	label_filtered=label[area>min_pix]
	area_filtered=area[area>min_pix]
	area_sort_ind=np.argsort(area_filtered)[::-1]
	label_filtered_sorted=label_filtered[area_sort_ind]
	area_filtered_sorted=area_filtered[area_sort_ind]
	#9,20
	com_bimg=np.logical_or((all_labels==label_filtered_sorted[0]),(all_labels==label_filtered_sorted[1]))
	
	mask=scipy.ndimage.morphology.binary_dilation(com_bimg,iterations=15).astype(np.int16)
	mask=np.logical_not(mask)
	dsum.close()
	
	mask_name=os.path.abspath(mask_name)
	m=h5py.File(mask_name,'w')
	m.create_dataset('/data/data',data=mask)
	m.close()

	plt.figure()
	plt.imshow(img_arry,cmap='jet')
	plt.title('The sum image')
	plt.clim(0,1e-3*img_arry.mean())
	plt.figure()
	
	plt.imshow(mask,cmap='jet')
	plt.title('The beam stop mask')


if __name__=='__main__':

	Eiger_file_name=sys.argv[1]
	mask_name=sys.argv[2]

	print('-----------\n Creating the beam stop mask, Please wait.')
	beam_stop_mask(Eiger_file_name,thld=100,mask_name='beam_stop_mask.h5') # parameters may be adjusted according to spcecific data.
	print('-----------\n ALL DONE!')

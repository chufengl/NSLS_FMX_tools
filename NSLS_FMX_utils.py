#!python -u

"""
Started on 3-8-2019 at NSLS 

Authors: Chufeng Li, Nadia Zatsepin

E-mail: chufengl@asu.edu

Usage:
 
	NSLS_FMX_utils.py <Eiger_file_name> <thld> <min_pix> <mask_file> <min_peak>

	thld: pixel value threshold
	min_pix: minimal number of pixels for a peak
	min_peak: minimal number of peaks for a hit
        
"""


import sys,os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, feature
import scipy
import glob
import h5py

def Eiger_img_read(Eiger_file_name,frame_no):
	'''
	This module reads the Eiger image file,with a given
	frame number.
	'''
	f=h5py.File(Eiger_file_name,'r')
	
	img_arry=np.array(f['entry/data/data'][frame_no,:,:],dtype='int32')
	print('The shape of the image is :',img_arry.shape)
	f.close()
	return img_arry

def single_peak_finder(img_arry,Eiger_file_name,frame_no,thld,min_pix,mask_file='None',interact=False):
	
	
	bimg=(img_arry>thld)
	
	if mask_file is not ' None':
		mask_file=os.path.abspath(mask_file)
		m=h5py.File(mask_file,'r')
		mask=m['/data/data'].value.astype(bool)
		m.close()
	elif mask_file is 'None':
		mask=np.ones_like(img_arry).astype(bool)
	else:
		sys.exit('the mask file option is inproper.')
	
	bimg=bimg*mask
	all_labels=measure.label(bimg)
	props=measure.regionprops(all_labels,img_arry)

	area=np.array([r.area for r in props]).reshape(-1,)
	max_intensity=np.array([r.max_intensity for r in props]).reshape(-1,)
	#coords=np.array([r.coords for r in props]).reshape(-1,)
	label=np.array([r.label for r in props]).reshape(-1,)
	centroid=np.array([np.array(r.centroid).reshape(1,2) for r in props]).reshape((-1,2))
	weighted_centroid=np.array([r.weighted_centroid for r in props]).reshape(-1,)
	label_filtered=label[area>min_pix]
	area_filtered=area[area>min_pix]
	area_sort_ind=np.argsort(area_filtered)[::-1]
	label_filtered_sorted=label_filtered[area_sort_ind]
	area_filtered_sorted=area_filtered[area_sort_ind]
	weighted_centroid_filtered=np.zeros((len(label_filtered_sorted),2))
	for index,value in enumerate(label_filtered_sorted):
        
        	weighted_centroid_filtered[index,:]=np.array(props[value-1].weighted_centroid)
#	print('In image: %s \n %5d peaks are found' %(img_file_name, len(label_filtered_sorted)))
	beam_center=np.array([1492.98,2163.41])
	
	if interact:
		plt.figure(figsize=(15,15))
		plt.imshow(img_arry*(mask.astype(np.int16)),cmap='jet')
		plt.colorbar()
	#	plt.clim(0,0.5*thld)
		plt.clim(0,100)
		plt.scatter(weighted_centroid_filtered[:,1],weighted_centroid_filtered[:,0],edgecolors='r',facecolors='none')
	#	plt.scatter(beam_center[1],beam_center[0],marker='*',color='b')
		title_Str=Eiger_file_name+'\nEvent: %d '%(frame_no)
		plt.title(title_Str)
		plt.show()
	return label_filtered_sorted,weighted_centroid_filtered,props 



def file_hit_finder(Eiger_file_name,thld,min_pix,min_peak,mask_file='None'):
	
	Eiger_file_name=os.path.abspath(Eiger_file_name)
	db=h5py.File(Eiger_file_name,'r')
	img_block=db['/entry/data/data'].value
	db.close()
	print('--------------')
	print(Eiger_file_name)
	print('the data block shape is: ',img_block.shape)
	
	if mask_file is not ' None':
		mask_file=os.path.abspath(mask_file)
		m=h5py.File(mask_file,'r')
		mask=m['/data/data'].value.astype(bool)
		m.close()
	elif mask_file is 'None':
		mask=np.ones_like(img_arry).astype(bool)
	else:
		sys.exit('the mask file option is inproper.')
	
	HIT_counter=0
	HIT_event_no_list=[]	

	for event_no in range(img_block.shape[0]):
		img_arry=img_block[event_no,:,:]
		bimg=(img_arry>thld)
		bimg=bimg*mask
		all_labels=measure.label(bimg)
		props=measure.regionprops(all_labels,img_arry)
		area=np.array([r.area for r in props]).reshape(-1,)
		max_intensity=np.array([r.max_intensity for r in props]).reshape(-1,)
		label=np.array([r.label for r in props]).reshape(-1,)
		centroid=np.array([np.array(r.centroid).reshape(1,2) for r in props]).reshape((-1,2))
		weighted_centroid=np.array([r.weighted_centroid for r in props]).reshape(-1,)
		label_filtered=label[area>min_pix]
		area_filtered=area[area>min_pix]
		area_sort_ind=np.argsort(area_filtered)[::-1]
		label_filtered_sorted=label_filtered[area_sort_ind]
		area_filtered_sorted=area_filtered[area_sort_ind]
		weighted_centroid_filtered=np.zeros((len(label_filtered_sorted),2))

		
		
		for index,value in enumerate(label_filtered_sorted):
			weighted_centroid_filtered[index,:]=np.array(props[value-1].weighted_centroid)
		
		peak_no=len(area_filtered_sorted)
		if peak_no>=min_peak:
			HIT_counter+=1
			HIT_event_no_list.append(event_no)
			
		
			print('HIT!!!!  %d  Event: %d peaks found'%(event_no,peak_no))
		else:
			print('BLANK!   %d  Event: %d peaks found'%(event_no,peak_no))
			#pass
	
	print(Eiger_file_name)
	print('%d   out of  %d  hits found!'%(HIT_counter,img_block.shape[0]))
	print('HIT rate: %.2f %%'%(100*HIT_counter/img_block.shape[0]))
	print('HIT events:')
	print(HIT_event_no_list)

	return None



#if __name__=='__main__':
	
#	Eiger_file_name=sys.argv[1]
#	frame_no=int(sys.argv[2])
#	thld=int(sys.argv[3])
#	min_pix=int(sys.argv[4])
#	mask_file=sys.argv[5]
#	img_show_flag=(sys.argv[6]=='True')
	
#	img_arry=Eiger_img_read(Eiger_file_name,frame_no)
#	label_filtered_sorted,weighted_centroid_filtered,props=peak_finder(img_arry,Eiger_file_name,frame_no,thld,min_pix,mask_file=mask_file,interact=img_show_flag)
	
#	print('In image: %s \n %5d peaks are found' %(Eiger_file_name, len(label_filtered_sorted)))



if __name__=='__main__':

	Eiger_file_name=sys.argv[1]
	thld=int(sys.argv[2])
	min_pix=int(sys.argv[3])
	mask_file=sys.argv[4]
	min_peak=int(sys.argv[5])
	file_hit_finder(Eiger_file_name,thld,min_pix,min_peak,mask_file=mask_file)


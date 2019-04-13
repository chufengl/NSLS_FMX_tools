#!python -u
"""
Started on 4-12-2019 at NSLS
Authors: Chufeng Li
E-mail: chufengl@asu.edu
Usage:

	CXI_hit_finder_EZ.py <CXI_file_name> <thld> <min_pix> <max_pix> <mask_file> <min_peak>
	thld: pixel value threshold
	min_pix: minimal number of pixels for a peak
    max_pix: maximal number of pixels for a peak
	mask_file: name of the mask file
	min_peak: minimal number of peaks for a hit

Parameter_tweaking mode:

label_filtered_sorted,weighted_centroid_filtered,props= \
single_peak_finder(CXI_file_name,event_no,thld,min_pix,max_pix,mask_file,interact='True')

"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, feature
import scipy
import glob
import h5py
import mpi4py.MPI as MPI
from pytictoc import TicToc
import pickle as pk

def CXI_img_read(CXI_file_name,frame_no):
	'''
	This module reads the CXI image file,with a given
	frame number.
	'''
	f=h5py.File(CXI_file_name,'r')

	img_arry=np.array(f['entry_1/data_1/data'][frame_no,:,:],dtype='int32')
	print('The shape of the image is :',img_arry.shape)
	f.close()
	return img_arry
def single_peak_finder(CXI_file_name,event_no,thld,min_pix,max_pix,mask_file,interact):

	img_arry=CXI_img_read(CXI_file_name,event_no)
	bimg=(img_arry>thld)

	if mask_file!='None':
		mask_file=os.path.abspath(mask_file)
		m=h5py.File(mask_file,'r')
		mask=m['/data/data'].value.astype(bool)
		m.close()
	elif mask_file=='None':
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
	label_filtered=label[(area>=min_pix)*(area<max_pix)]
	area_filtered=area[(area>=min_pix)*(area<max_pix)]
	area_sort_ind=np.argsort(area_filtered)[::-1]
	label_filtered_sorted=label_filtered[area_sort_ind]
	area_filtered_sorted=area_filtered[area_sort_ind]
	weighted_centroid_filtered=np.zeros((len(label_filtered_sorted),2))
	for index,value in enumerate(label_filtered_sorted):

        	weighted_centroid_filtered[index,:]=np.array(props[value-1].weighted_centroid)
#	print('In image: %s \n %5d peaks are found' %(img_file_name, len(label_filtered_sorted)))
	beam_center=np.array([1492.98,2163.41])

	if interact=='True':
		plt.figure(figsize=(15,15))
		plt.imshow(img_arry*(mask.astype(np.int16)),cmap='gray')
		plt.colorbar()
	#	plt.clim(0,0.5*thld)
		plt.clim(0,(img_arry*(mask.astype(np.int16))).mean()*2)
		plt.scatter(weighted_centroid_filtered[:,1],weighted_centroid_filtered[:,0],edgecolors='r',facecolors='none')
	#	plt.scatter(beam_center[1],beam_center[0],marker='*',color='b')
		title_Str=CXI_file_name+'\nEvent: %d '%(event_no)
		plt.title(title_Str)
		plt.show()
	return label_filtered_sorted,weighted_centroid_filtered,props



def file_hit_finder(CXI_file_name,thld,min_pix,max_pix,min_peak,mask_file='None',Region='ALL'):



    if Region=='ALL':
        x_min=0
        y_min=0
        x_max=4371
        y_max=4150
    elif Region=='Q':
        x_min=0
        y_min=0
        x_max=4371
        y_max=4150
    elif Region=='C':
        x_min=0
        y_min=0
        x_max=4371
        y_max=4150
    else:
        sys.exit('Check the Region option: ALL,Q,C')
    CXI_file_name=os.path.abspath(CXI_file_name)
    db=h5py.File(CXI_file_name,'r')
    img_block=db['/entry_1/data_1/data'][:,x_min:x_max,y_min:y_max]

    db.close()
    print('--------------')
    print(CXI_file_name)
    print('the data block shape is: ',img_block.shape)

    if mask_file!='None':
        mask_file=os.path.abspath(mask_file)
        m=h5py.File(mask_file,'r')
        mask=m['/data/data'][x_min:x_max,y_min:y_max].astype(bool)
        m.close()
    elif mask_file=='None':
        mask=np.ones((img_block.shape[1],img_block.shape[2])).astype(bool)
    else:
        sys.exit('the mask file option is inproper.')
    HIT_counter=0
    HIT_event_no_list=[]
    peakXPosRaw=np.zeros((img_block.shape[0],1024))
    peakYPosRaw=np.zeros((img_block.shape[0],1024))
    pixel_size=np.float(110e-6)
    nPeaks=np.zeros((img_block.shape[0],),dtype=np.int16)
    peakTotalIntensity=np.zeros((img_block.shape[0],1024))
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
        weighted_centroid=np.array([r.weighted_centroid for r in props]).reshape(-1,2)
        label_filtered=label[(area>=min_pix)*(area<max_pix)]
        area_filtered=area[(area>=min_pix)*(area<max_pix)]
        area_sort_ind=np.argsort(area_filtered)[::-1]
        label_filtered_sorted=label_filtered[area_sort_ind]
        area_filtered_sorted=area_filtered[area_sort_ind]
        weighted_centroid_filtered=np.zeros((len(label_filtered_sorted),2))
        max_intensity_filtered=np.zeros((len(label_filtered_sorted),1))
        mean_intensity_filtered=np.zeros((len(label_filtered_sorted),1))
        for index,value in enumerate(label_filtered_sorted):
            weighted_centroid_filtered[index,:]=np.array(props[value-1].weighted_centroid)
            max_intensity_filtered[index,:]=props[value-1].max_intensity
            mean_intensity_filtered[index,:]=props[value-1].mean_intensity

            peak_no=len(area_filtered_sorted)
            nPeaks[event_no]=np.int16(np.minimum(1024,peak_no))
            # print(nPeaks[event_no])
            # print(peakTotalIntensity.shape)
            # print(mean_intensity_filtered.shape)
            peakTotalIntensity[event_no,:nPeaks[event_no]]=mean_intensity_filtered.reshape(-1,)
            peakXPosRaw[event_no,:nPeaks[event_no]]=weighted_centroid_filtered[:,0]
            peakYPosRaw[event_no,:nPeaks[event_no]]=weighted_centroid_filtered[:,1]

        if (peak_no>=min_peak) and (peak_no<=1024):
            HIT_counter+=1
            HIT_event_no_list.append(event_no)


            print('HIT!!!!  %d  Event: %d peaks found'%(event_no,peak_no))
        else:
            print('BLANK!   %d  Event: %d peaks found'%(event_no,peak_no))
            #pass


    peak_list_dict={'CXI_file_name':CXI_file_name,'nPeaks':nPeaks.astype(np.int16),'peakTotalIntensity':peakTotalIntensity,\
            'peakXPosRaw':peakXPosRaw,'peakYPosRaw':peakYPosRaw}
    pf=open(os.path.split(CXI_file_name)[1]+'_peak_info.pkl','wb')
    pk.dump(peak_list_dict,pf)
    pf.close()
    print(CXI_file_name)
    print('%d   out of  %d  hits found!'%(HIT_counter,img_block.shape[0]))
    print('HIT rate: %.2f %%'%(100*HIT_counter/img_block.shape[0]))
    print('HIT events:')
    print(HIT_event_no_list)
    print('Peak information saved in '+os.path.split(CXI_file_name)[1]+'_peak_info.pkl')

	#lf=open(os.path.split(CXI_file_name)[1]+'HIT.log','w')
	#lf.write('%s'%(CXI_file_name))
	#lf.write('\n %d   out of  %d  hits found!'%(HIT_counter,img_block.shape[0]))
	#lf.write('\n HIT rate: %.2f %%'%(100*HIT_counter/img_block.shape[0]))
	#lf.write('\n HIT events:\n')
	#for event in HIT_event_no_list:
		#lf.write('%d \n'%event)
	#lf.write('-----------------')
	#lf.close()

    total_event_no=img_block.shape[0]
    hit_rate=100*HIT_counter/total_event_no



    return total_event_no, HIT_counter, hit_rate, HIT_event_no_list



if __name__=='__main__':
    print(__doc__)
    t = TicToc()
    t.tic()

    CXI_file_name=sys.argv[1]
    CXI_file_name=os.path.abspath(CXI_file_name)
    thld=int(sys.argv[2])
    min_pix=int(sys.argv[3])
    max_pix=int(sys.argv[4])
    mask_file=sys.argv[5]
    min_peak=int(sys.argv[6])
    Region=sys.argv[7]
    file_hit_finder(CXI_file_name,thld,min_pix,max_pix,min_peak,mask_file=mask_file,Region=Region)
    t.toc(CXI_file_name+'\n took ')

#!python -u

"""
Started on 3-8-2019 at NSLS
Authors: Chufeng Li, Nadia Zatsepin
E-mail: chufengl@asu.edu
Usage:

	NSLS_FMX_utils.py <Eiger_file_name> <thld> <min_pix> <max_pix> <mask_file> <min_peak> <Region>
	thld: pixel value threshold
	min_pix: minimal number of pixels for a peak
	min_peak: minimal number of peaks for a hit
	Region: 'ALL', 'C', 'Q'

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
#import fabio

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



def file_hit_finder_subset(Eiger_file_name,rank,size,skip,thld,min_pix,max_pix,min_peak,mask_file='None',Region='ALL'):
	if Region=='ALL':
		x_min=0
		y_min=0
		x_max=4371
		y_max=4150
	elif Region=='Q':
		x_min=2185
		y_min=2075
		x_max=4370
		y_max=4150
	elif Region=='C':
		x_min=1092
		y_min=1037
		x_max=3278
		y_max=3112
	else:
		sys.exit('Check the Region option: ALL,Q,C')
	Eiger_file_name=os.path.abspath(Eiger_file_name)
	db=h5py.File(Eiger_file_name,'r') #h5py read
	db_1=db['/entry/data/data']       #h5py read
	#db=fabio.open(Eiger_file_name)     #fabio read
 	#db_1=db.dataset[0]                 #fabio read
	job_total_load=db_1.shape[0]
	job_per_load=np.ceil(job_total_load/size).astype(np.int16)
	frame_list=np.arange(rank*job_per_load,np.minimum((rank+1)*job_per_load,job_total_load),skip)
	img_block=np.array(db_1[frame_list,x_min:x_max,y_min:y_max])  #h5py read
	db.close()   #h5py read

	total_event_no=len(frame_list)

	print('--------------')
	print(Eiger_file_name)
	print('the data block shape is: ',img_block.shape)  #h5py read
	#print('the data block shape is: (%d,%d,%d)'%(len(frame_list),db_1.shape[1],db_1.shape[2])) #fabio read
	if mask_file!='None':
		mask_file=os.path.abspath(mask_file)
		m=h5py.File(mask_file,'r')
		mask=m['/data/data'][x_min:x_max,y_min:y_max].astype(bool)
		m.close()
	elif mask_file=='None':
		mask=np.ones_like(img_block[0,:,:]).astype(bool)  #h5py read
		#mask=np.ones_like(db.data).astype(bool)            #fabio read
	else:
		sys.exit('the mask file option is inproper.')
	HIT_counter=0
	HIT_event_no_list=[]
	peakXPosRaw=np.zeros((len(frame_list),1024))
	peakYPosRaw=np.zeros((len(frame_list),1024))
	pixel_size=np.float(110e-6)#to be changed
	nPeaks=np.zeros((len(frame_list),),dtype=np.int16)
	peakTotalIntensity=np.zeros((len(frame_list),1024))

	for l in range(len(frame_list)):
		event_no=frame_list[l]
		img_arry=img_block[l,:,:]  #h5py read
		#db.get_frame(event_no)     #fabio read
		#img_arry=np.array(db.data) #fabio read
		bimg=(img_arry>thld)
		bimg=bimg*mask
		all_labels=measure.label(bimg)
		props=measure.regionprops(all_labels,img_arry)
		area=np.array([r.area for r in props]).reshape(-1,)
		max_intensity=np.array([r.max_intensity for r in props]).reshape(-1,)
		label=np.array([r.label for r in props]).reshape(-1,)
		centroid=np.array([np.array(r.centroid).reshape(1,2) for r in props]).reshape((-1,2))
		weighted_centroid=np.array([r.weighted_centroid for r in props]).reshape(-1,)
		label_filtered=label[(area>=min_pix)*(area<max_pix)]
		area_filtered=area[(area>=min_pix)*(area<max_pix)]
		area_sort_ind=np.argsort(area_filtered)[::-1]
		label_filtered_sorted=label_filtered[area_sort_ind]
		area_filtered_sorted=area_filtered[area_sort_ind]
		weighted_centroid_filtered=np.zeros((len(label_filtered_sorted),2))
		max_intensity_filtered=np.zeros((len(label_filtered_sorted),1))
		mean_intensity_filtered=np.zeros((len(label_filtered_sorted),1))

		peak_no=len(area_filtered_sorted)
		nPeaks[l]=np.int16(np.minimum(1024,peak_no))
		for index,value in enumerate(label_filtered_sorted):
			weighted_centroid_filtered[index,:]=np.array(props[value-1].weighted_centroid)
			max_intensity_filtered[index,:]=props[value-1].max_intensity
			mean_intensity_filtered[index,:]=props[value-1].mean_intensity

			peakTotalIntensity[l,:nPeaks[l]]=mean_intensity_filtered[:nPeaks[l],0].reshape(-1,)
			peakXPosRaw[l,:nPeaks[l]]=weighted_centroid_filtered[:nPeaks[l],0]+x_min
			peakYPosRaw[l,:nPeaks[l]]=weighted_centroid_filtered[:nPeaks[l],1]+y_min
        #peak_no=len(area_filtered_sorted)
		if (peak_no>=min_peak) and (peak_no<=1024):
			HIT_counter+=1
			HIT_event_no_list.append(event_no)
			print('HIT!!!!  %d  Event: %d peaks found'%(event_no,peak_no))
		else:
			print('BLANK!   %d  Event: %d peaks found'%(event_no,peak_no))
            #pass
	peak_list_dict={'Eiger_file_name':Eiger_file_name,'frame_list':frame_list,\
	'HIT_event_no_list':HIT_event_no_list,'nPeaks':nPeaks.astype(np.int16),\
	'peakTotalIntensity':peakTotalIntensity,\
	'peakXPosRaw':peakXPosRaw,'peakYPosRaw':peakYPosRaw}
	print(Eiger_file_name)
	print('%d   out of  %d  hits found!'%(HIT_counter,img_block.shape[0]))
	print('HIT rate: %.2f %%'%(100*HIT_counter/img_block.shape[0]))
	print('HIT events:')
	print(HIT_event_no_list)

	#lf=open(os.path.split(Eiger_file_name)[1]+'HIT.log','w')
	#lf.write('%s'%(Eiger_file_name))
	#lf.write('\n %d   out of  %d  hits found!'%(HIT_counter,img_block.shape[0]))
	#lf.write('\n HIT rate: %.2f %%'%(100*HIT_counter/img_block.shape[0]))
	#lf.write('\n HIT events:\n')
	#for event in HIT_event_no_list:
		#lf.write('%d \n'%event)
	#lf.write('-----------------')
	#lf.close()
	#total_event_no=img_block.shape[0]
	hit_rate=100*HIT_counter/total_event_no
	return total_event_no, HIT_counter, hit_rate, HIT_event_no_list, peak_list_dict

def Eiger_file_list(find_list_file):
	find_list_file=os.path.abspath(find_list_file)
	l=open(find_list_file,'r')
	list_s=l.readlines()

	return list_s

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
	t = TicToc()
	t.tic()

	comm=MPI.COMM_WORLD
	size=comm.Get_size()
	rank=comm.Get_rank()
	print('I am rank %d of %d'%(rank,size))
	find_list_file=sys.argv[1]
	find_list_file=os.path.abspath(find_list_file)
	list_s=Eiger_file_list(find_list_file)
    #Eiger_file_name=sys.argv[1]
	thld=int(sys.argv[2])
	min_pix=int(sys.argv[3])
	max_pix=int(sys.argv[4])
	mask_file=sys.argv[5]
	min_peak=int(sys.argv[6])
	Region=sys.argv[7]
	skip=1
	lf=open(os.path.split(find_list_file)[1]+'HIT-rank%d.log'%(rank),'w',1)
	lf.write('Eiger file list: %s\n'%(find_list_file))
	lf.write('thld: %d\n'%(thld))
	lf.write('min_pix: %d\n'%(min_pix))
	lf.write('max_pix: %d\n'%(max_pix))
	lf.write('min_peak: %d\n'%(min_peak))
	lf.write('mask_file: %s\n'%(mask_file))
	ef=open(os.path.split(find_list_file)[1]+'eve-rank%d.lst'%(rank),'w',1)
	pf=open(os.path.split(find_list_file)[1]+'-rank%d.pk'%(rank),'w',1)
	pf_header='Eiger_file_name event_no hit_tag peak_no peak_id peak_x peak_y peak_Int\n'

	pf.write(pf_header)

	for l in range(len(list_s)):
		Eiger_file_name=list_s[l][:-1]
		print('hit finding %d file  out of %d    \n%s'%(l+1,len(list_s),list_s[l]))
		total_event_no, HIT_counter, hit_rate,HIT_event_no_list,peak_list_dict=\
		file_hit_finder_subset(list_s[l][:-1],rank,size,skip,thld,min_pix,max_pix,min_peak,mask_file=mask_file,Region=Region)
		lf.write('%s'%(Eiger_file_name))
		lf.write('\n %d   out of  %d  hits found!'%(HIT_counter,total_event_no))
		lf.write('\n HIT rate: %.2f %%\n'%(hit_rate))
		for event in HIT_event_no_list:
			ef.write('%s //%d\n'%(Eiger_file_name,event))
		for l in range(len(peak_list_dict['frame_list'])):
			event=peak_list_dict['frame_list'][l]
			hit_tag=int(event in HIT_event_no_list)
			for peak_id in range(peak_list_dict['nPeaks'][l]):
				pf.write('%s %d %d %d %d %.2f %.2f %.2f\n'\
				%(peak_list_dict['Eiger_file_name'],event,hit_tag,peak_list_dict['nPeaks'][l],peak_id,\
				peak_list_dict['peakXPosRaw'][l][peak_id],peak_list_dict['peakYPosRaw'][l][peak_id],peak_list_dict['peakTotalIntensity'][l][peak_id]))
	lf.close()
	ef.close()
	pf.close()

	t.toc('Worker %d took'%(rank))
	print('!!!!ALL DONE!!!')

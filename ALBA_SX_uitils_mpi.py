"""
Started on 4-20-2019
Authors: Chufeng Li, Nadia Zatsepin
E-mail: chufengl@asu.edu
Usage:
    ALBA_SX_uitils.py <cbf_file_list_file> <thld> <min_pix> <max_pix> <mask_file> <min_peak> <Region>
    thld: pixel value threshold
    min_pix: minimal number of pixels for a peak
    max_pix: maximal number of pixels for a peak
    mask_file: name of the mask file
    min_peak: minimal number of peaks for a hit
    Region: 'ALL', 'C', 'Q'

Parameter_tweaking mode:

label_filtered_sorted,weighted_centroid_filtered,props= \
single_peak_finder(CBF_file_name,thld,min_pix,max_pix,mask_file,interact='True')

"""
import sys,os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, feature
import scipy
import glob
import h5py
import fabio
import mpi4py.MPI as MPI
from pytictoc import TicToc
import pickle as pk

def CBF_img_read(CBF_file_name):
    '''
    This module reads the cbf image file,with a given
    frame number.
    '''
    f=fabio.open(CBF_file_name)
    img_arry=np.array(f.data,dtype='int32')
    #print('The shape of the image is :',img_arry.shape)
    return img_arry
def single_peak_finder(CBF_file_name,thld,min_pix,max_pix,mask_file,interact):

    img_arry=CBF_img_read(CBF_file_name)
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
        # beam_center=np.array([1492.98,2163.41])
    if interact=='True':
        plt.figure(figsize=(15,15))
        plt.imshow(img_arry*(mask.astype(np.int16)),cmap='gray')
        #	plt.clim(0,0.5*thld)
        plt.clim(0,(img_arry*(mask.astype(np.int16))).mean()*20)
        plt.colorbar()
        plt.scatter(weighted_centroid_filtered[:,1],weighted_centroid_filtered[:,0],edgecolors='r',facecolors='none')
        #	plt.scatter(beam_center[1],beam_center[0],marker='*',color='b')
        title_Str=CBF_file_name
        plt.title(title_Str)
        plt.show()
    return label_filtered_sorted,weighted_centroid_filtered,props

def CBF_file_list(find_list_file):
	find_list_file=os.path.abspath(find_list_file)
	l=open(find_list_file,'r')
	list_s=l.readlines()

	return list_s


def List_hit_finder(cbf_file_list_file,thld,min_pix,max_pix,min_peak,mask_file='None',Region='ALL'):

    cbf_file_list_file=os.path.abspath(cbf_file_list_file)
    list_s=CBF_file_list(cbf_file_list_file)
    if len(list_s)==0:
        sys.exit('No CBF files have been found in :\n',cbf_file_list_file)
    img_arry_s=CBF_img_read(list_s[0][:-1])

    if Region=='ALL':
        x_min=0
        y_min=0
        x_max=img_arry_s.shape[0]
        y_max=img_arry_s.shape[1]
    elif Region=='Q':
        x_min=np.round(0.5*img_arry_s.shape[0]).astype(int)
        y_min=np.round(0.5*img_arry_s.shape[1]).astype(int)
        x_max=img_arry_s.shape[0]
        y_max=img_arry_s.shape[1]
    elif Region=='C':
        x_min=np.round(0.25*img_arry_s.shape[0]).astype(int)
        y_min=np.round(0.25*img_arry_s.shape[1]).astype(int)
        x_max=np.round(0.75*img_arry_s.shape[0]).astype(int)
        y_max=np.round(0.75*img_arry_s.shape[1]).astype(int)
    else:
        sys.exit('Check the Region option: ALL,Q,C')


    if mask_file!='None':
        mask_file=os.path.abspath(mask_file)
        m=h5py.File(mask_file,'r')
        mask=m['/data/data'][x_min:x_max,y_min:y_max].astype(bool)
        m.close()
    elif mask_file=='None':
        mask=np.ones((img_arry_s.shape[0],img_arry_s.shape[1])).astype(bool)
        mask=mask[x_min:x_max,y_min:y_max]
    else:
        sys.exit('the mask file option is inproper.')
    HIT_counter=0
    HIT_event_no_list=[]
    peakXPosRaw=np.zeros((len(list_s),1024))
    peakYPosRaw=np.zeros((len(list_s),1024))
    pixel_size=np.float(110e-6)#to be changed
    nPeaks=np.zeros((len(list_s),),dtype=np.int16)
    peakTotalIntensity=np.zeros((len(list_s),1024))

    lf=open(os.path.split(cbf_file_list_file)[1]+'HIT.log','w',1)
    ef=open(os.path.split(cbf_file_list_file)[1]+'eve.lst','w',1)

    lf.write('%s'%(cbf_file_list_file))
    lf.write('\n-----------')
    lf.write('\nthld: %d\nmin_pix: %d\nmax_pix: %d\nmin_peak: %d\nmask_file: %s\nRegion: %s'%\
    (thld,min_pix,max_pix,min_peak,mask_file,Region))
    lf.write('\n-----------')

    for event_no in range(len(list_s)):
        img_arry=CBF_img_read(list_s[event_no][:-1])
        img_arry=img_arry[x_min:x_max,y_min:y_max]
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
        peak_no=0
        for index,value in enumerate(label_filtered_sorted):
            weighted_centroid_filtered[index,:]=np.array(props[value-1].weighted_centroid)
            max_intensity_filtered[index,:]=props[value-1].max_intensity
            mean_intensity_filtered[index,:]=props[value-1].mean_intensity
            peak_no=len(area_filtered_sorted)
            nPeaks[event_no]=np.int16(np.minimum(1024,peak_no))
            # print(nPeaks[event_no])
            # print(peakTotalIntensity.shape)
            # print(mean_intensity_filtered.shape)
            peakTotalIntensity[event_no,:nPeaks[event_no]]=mean_intensity_filtered[:nPeaks[event_no],0].reshape(-1,)
            peakXPosRaw[event_no,:nPeaks[event_no]]=weighted_centroid_filtered[:nPeaks[event_no],0]+x_min
            peakYPosRaw[event_no,:nPeaks[event_no]]=weighted_centroid_filtered[:nPeaks[event_no],1]+y_min


        if (peak_no>=min_peak) and (peak_no<=1024):
            HIT_counter+=1
            HIT_event_no_list.append(event_no)


            print('HIT!!!!  Event %d: %d peaks found'%(event_no,peak_no))
            lf.write('\nHIT!!!!  Event %d: %d peaks found'%(event_no,peak_no))
            ef.write('%s\n'%(list_s[event_no][:-1]))
        else:
            print('BLANK!   Event %d: %d peaks found'%(event_no,peak_no))
            lf.write('\nBLANK!   Event %d: %d peaks found'%(event_no,peak_no))
            #pass


    peak_list_dict={'cbf_file_list_file':cbf_file_list_file,'nPeaks':nPeaks.astype(np.int16),'peakTotalIntensity':peakTotalIntensity,\
            'peakXPosRaw':peakXPosRaw,'peakYPosRaw':peakYPosRaw}
    pf=open(os.path.split(cbf_file_list_file)[1]+'_peak_info.pkl','wb')
    pk.dump(peak_list_dict,pf)
    pf.close()
    print(cbf_file_list_file)
    print('%d   out of  %d  hits found!'%(HIT_counter,len(list_s)))
    print('HIT rate: %.2f %%'%(100*HIT_counter/len(list_s)))
    print('HIT events:')
    print(HIT_event_no_list)
    print('Peak information saved in '+os.path.split(cbf_file_list_file)[1]+'_peak_info.pkl')

    total_event_no=len(list_s)
    hit_rate=100*HIT_counter/total_event_no


    lf.write('\n------------------------------------------------------------------')
    lf.write('\n%s'%(cbf_file_list_file))
    lf.write('\n %d   out of  %d  hits found!'%(HIT_counter,total_event_no))
    lf.write('\n HIT rate: %.2f %%'%(hit_rate))
    #lf.write('\n HIT events:\n')
    lf.write('\n------------------------------------------------------------------')
    #for event in HIT_event_no_list:
    #lf.write('%d \n'%event)
    #lf.write('-----------------')
    lf.close()
    ef.close()

    return total_event_no, HIT_counter, hit_rate, HIT_event_no_list

if __name__=='__main__':
    print(__doc__)
    t = TicToc()
    t.tic()

    cbf_file_list_file=sys.argv[1]
    cbf_file_list_file=os.path.abspath(cbf_file_list_file)
    thld=int(sys.argv[2])
    min_pix=int(sys.argv[3])
    max_pix=int(sys.argv[4])
    mask_file=sys.argv[5]
    min_peak=int(sys.argv[6])
    Region=sys.argv[7]


    List_hit_finder(cbf_file_list_file,thld,min_pix,max_pix,min_peak,mask_file=mask_file,Region=Region)
    t.toc(cbf_file_list_file+'\n took ')

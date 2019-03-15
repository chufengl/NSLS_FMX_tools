#!/GPFS/CENTRAL/xf17id1/mfuchs/CFL/Anaconda3/bin/python -u

"""
Usage:

	python NSLS_convters.py  <master_file_name> <Dozor_dat_name> <output_dat_name> [options]

Options:
	<master_file_name>	master file name for a data set
	<Dozor_dat_name>	input .dat file name from Dozor
	<output_dat_name>	output .dat file name for DatView
	-h --help 		show this message



#---------------------------

#Authors: Chufeng Li,
        #Nadia Zatsepin,

#Department of Physics
#Arizona State University


#E-mail: chufengl@asu.edu


#Started 3-8-2019 @ NSLS

#----------------------------

"""


#%matplotlib tk
import sys,os
sys.path.append('/GPFS/CENTRAL/xf17id1/mfuchs/CFL/dev')
import numpy as np
import matplotlib.pyplot as plt
import h5py
from docopt import docopt

def old_dat_read(od_file_name):
	od_file_name=os.path.abspath(od_file_name)
	f=open(od_file_name,'r')
#	od_arry=np.loadtxt(od_file_name,skiprows=7)
	str_list=f.readlines()	
	f.close()
	od_arry=-1*np.ones((len(str_list)-7,12))
	row_counter=0
	for line in str_list[7:]:
		s=line.split('|',4)
		s0=s[0]
		s1=s[1]
		s2=s[2]
		s3=s[3]
		n0=int(s0)
		n1=[]
		n2=[]
		n3=[]
		s1_list=s1.split(' ',20)
	#	print('s1_list',s1_list)
		for ch1 in s1_list:
			if (ch1 is not ' ') and (ch1 is not ''):
				n1.append(ch1)

		if not s2.startswith(' -----'):
			s2_list=s2.split(' ',20)
			for ch2 in s2_list:
				if (ch2 is not ' ') and (ch2 is not ''):
					n2.append(ch2)
		else:
			n2=[-1]*5

		s3_list=s3.split(' ',20)
		for ch3 in s3_list:
			if (ch3 is not ' ') and (ch3 is not ''):
				n3.append(ch3)

		od_arry[row_counter,0]=n0
		od_arry[row_counter,1:4]=np.array(n1)
		od_arry[row_counter,4:9]=np.array(n2)
		od_arry[row_counter,9:12]=np.array(n3)

		row_counter=row_counter+1
		
	
	return od_arry


def master_read(master_file_name):

	master_file_name=os.path.abspath(master_file_name)

	base_name=(os.path.split(master_file_name))[1].split('master')[0]
	m=h5py.File(master_file_name,'r')
	file_list=list(m['/entry/data'])
	#print('file_list:',file_list)
	file_list.sort()
	#print('sorted:-------',file_list)
	file_dir=os.path.dirname(master_file_name)
	data_full_file_name_list=[]
	size_list=[]
	wave_length=m['/entry/instrument/beam/incident_wavelength'].value
	beam_center_x=m['/entry/instrument/detector/beam_center_x'].value
	beam_center_y=m['/entry/instrument/detector/beam_center_y'].value
	data_col_date=m['/entry/instrument/detector/detectorSpecific/data_collection_date'].value
	det_distance=m['entry/instrument/detector/detector_distance'].value
	
	m.close()
	for file in file_list:
		file_name=base_name+file+'.h5'
		file_name=file_dir+'/'+file_name
		data_full_file_name_list.append(file_name)
		d=h5py.File(file_name,'r')
		size=d['/entry/data/data'].shape
		size_list.append(size)
		d.close()
	master_dict={'data_file_name_list':data_full_file_name_list,'size_list':size_list,'wave_length':wave_length,\
			'beam_center_x':beam_center_x,'beam_center_y':beam_center_y,'det_distance':det_distance,'data_col_date':data_col_date}
	return master_dict


def od2ne(master_dict,od_arry):
	n_images_pfile=master_dict['size_list'][0][0]
	n_files=len(master_dict['data_file_name_list'])
	event_no_arry=od_arry[:,0]-1
	event_no_arry=np.mod(event_no_arry,n_images_pfile)
	ne_arry=np.concatenate((event_no_arry.reshape((-1,1)),od_arry[:,1:]),axis=-1)
	
	return ne_arry



def dat_write(master_dict,dat_file_name,ne_arry):

	dat_file_name=os.path.abspath(dat_file_name)
	df=open(dat_file_name,'w')
	
	n_images_pfile=master_dict['size_list'][0][0]
	file_name_list=master_dict['data_file_name_list']
	
	file_name_list_pad=[]
	
	for file in file_name_list:
		file_name_list_pad=file_name_list_pad+[file]*n_images_pfile
	print(ne_arry.shape[0])	
	file_name_list_pad=file_name_list_pad[0:ne_arry.shape[0]]
	
	cname0='ifile'
	cname1='event'
	cname2='no_of_spots'
	cname3='average_Int'
	cname4='Resolution'
	cname5='Scale_fac'
	cname6='B-fac'
	cname7='Corr.'
	cname8='R-fac'
	cname9='Main_Score'
	cname10='Spot_Score'
	cname11='Visible_Res'
	df.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(cname0,cname1,cname2,cname3,cname4,cname5,cname6,cname7,cname8,cname9,cname10,cname11))	
	for r in range(ne_arry.shape[0]):
		df.write('%s\t'%(file_name_list_pad[r]))
		df.write('%d\t%d\t%d\t%.1f\t%.2f\t%.1f\t%.1f\t%.1f\t%.1f\t%.3f\t%.2f\t%.2f\n'%(ne_arry[r,0],ne_arry[r,1],ne_arry[r,2]\
				,ne_arry[r,3],ne_arry[r,4],ne_arry[r,5],ne_arry[r,6],ne_arry[r,7],ne_arry[r,8],ne_arry[r,9]\
				,ne_arry[r,10],ne_arry[r,11]))
	df.close()
	print('file writing done!----------')

if __name__=='__main__':
	#print('executing script mode!')
	#argv=docopt(__doc__)
	print(__doc__)
	#print('ready to print argv')
	#argv=docopt(__doc__)
	#print(argv)
	#master_file_name=argv['<master_file_name>']
	#od_file_name=argv['<Dozor_dat_name>']
	#dat_file_name=argv['<output_dat_name>']
	
	master_file_name=sys.argv[1]
	od_file_name=sys.argv[2]
	dat_file_name=sys.argv[3]	

	master_dict=master_read(master_file_name)
	od_arry=old_dat_read(od_file_name)
	ne_arry=od2ne(master_dict,od_arry)
	dat_write(master_dict,dat_file_name,ne_arry)
	
	
	

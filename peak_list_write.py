'''
Started on 4-12-2019 at NSLS
Authors: Chufeng Li
E-mail: chufengl@asu.edu

write peak lists form CXI_hit_finder_EZ to cxi file,
according to the Cheetah and CrystFEL default.

Usage:

    python peak_list_write.py <w_CXI_file_name> <pkl_file_name>

w_CXI_file_name:  the cxi file to which the peak list is written to,
                    usually, this should be a copy of the raw file.
pkl_file_name:    the .pkl file generated from CXI_hit_finder_EZ.py
'''
import sys,os
import h5py
import numpy as np
import pickle as pk

def peak_list_write_func(w_CXI_file_name,pkl_file_name):
    f=open(pkl_file_name,'rb')
    peak_dict=pk.load(f)
    f.close()
    print('This peak list was from: \n',peak_dict['CXI_file_name'])
    peak_h5_path_prefix='/entry_1/CFL_peaks/'
    d=h5py.File(w_CXI_file_name,'a')
    d.create_dataset(peak_h5_path_prefix+'nPeaks',data=peak_dict['nPeaks'])
    d.create_dataset(peak_h5_path_prefix+'peakTotalIntensity',data=peak_dict['peakTotalIntensity'])
    d.create_dataset(peak_h5_path_prefix+'peakXPosRaw',data=peak_dict['peakXPosRaw'])
    d.create_dataset(peak_h5_path_prefix+'peakYPosRaw',data=peak_dict['peakYPosRaw'])
    d.close()
    print('peak_lists writting completed!')
    return None
if __name__=='__main__':
    w_CXI_file_name=sys.argv[1]
    w_CXI_file_name=os.path.abspath(w_CXI_file_name)
    pkl_file_name=sys.argv[2]
    pkl_file_name=os.path.abspath(pkl_file_name)
    peak_list_write_func(w_CXI_file_name,pkl_file_name)

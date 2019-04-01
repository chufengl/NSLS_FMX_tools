'''
comput_info.py comstructs the Obeject Orientation framework to extract the
computation information.

Usage：
    python comput_info.py <path> <key_word>

<path>:         the path to the computation trial folder_list
<key_word>:     the key_word of the trial folders to be used with the wildcards.

e.g. :  python comput_info.py . core

'''

import sys,os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

class Folder(object):
    msg='This is a Folder, with log and result files inside.'

    def __init__(self,folder_name):

        self.folder_name=os.path.abspath(folder_name)


    @property
    def out_file_name(self):

        file=glob.glob(self.folder_name+'/*.out')
        file.sort()
        if len(file)==0:
            sys.exit('No .out file found in \n%s'%(self.folder_name))
        else:
            out_file_name=file[0]
        return out_file_name

    @property
    def comp_time_lst(self):
        f_out=open(self.out_file_name,'r')
        txt=f_out.readlines()
        f_out.close()
        txts=str([])
        for l in txt:
            txts=txts+l

        lines=re.findall('Worker .* took .*',txts)

        comp_time_lst=[[],[]]
        for entry in lines:
            t=re.findall('[-+]?\d+\.?\d{0,8}',entry)
            comp_time_lst[0].append(int(t[0]))
            comp_time_lst[1].append(float(t[1]))
        size=len(comp_time_lst[0])
        return {'num_workers':size,'computation_time_list':comp_time_lst}
    @property
    def comp_time_mean(self):
        num_cores=self.comp_time_lst['num_workers']
        comp_time_mean=np.array(self.comp_time_lst['computation_time_list'][1]).mean()
        return comp_time_mean
    @property
    def comp_time_min(self):
        num_cores=self.comp_time_lst['num_workers']
        comp_time_min=np.array(self.comp_time_lst['computation_time_list'][1]).min()
        return comp_time_min
    @property
    def comp_time_max(self):
        num_cores=self.comp_time_lst['num_workers']
        comp_time_max=np.array(self.comp_time_lst['computation_time_list'][1]).max()
        return comp_time_max



if __name__=='__main__':
    path=os.path.abspath(sys.argv[1])
    key_word=sys.argv[2]
    folder_list=glob.glob(path+'/*'+key_word+'*')
    num_cores=[]
    mean_time=[]
    max_time=[]
    min_time=[]
    for folder_name in folder_list:
        short_name=os.path.basename(folder_name)
        vars()[short_name+'_obj']=Folder(folder_name)
        num_cores.append(vars()[short_name+'_obj'].comp_time_lst['num_workers'])
        mean_time.append(vars()[short_name+'_obj'].comp_time_mean)
        max_time.append(vars()[short_name+'_obj'].comp_time_max)
        min_time.append(vars()[short_name+'_obj'].comp_time_min)

    plt.figure(1)
    plt.title('NSLS_hit_finding computation time plot')
    plt.plot(num_cores,mean_time,'r+',num_cores,max_time,'gs',num_cores,min_time,'b^')
    plt.xlabel('number of cores used for computation')
    plt.ylabel('computation time of all cores')
    leg_label=['mean_time','max_time','min_time']
    plt.legend(leg_label,loc='upper right')
    plt.xticks(np.arange(0,110,step=10))
    plt.show()
    plt.savefig('comp_time.png')

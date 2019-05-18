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
        re.findall('Eiger file list: .*',txts)

        comp_time_lst=[[],[]]
        for entry in lines:
            t=re.findall('[-+]?\d+\.?\d{0,8}',entry)
            comp_time_lst[0].append(int(t[0]))
            comp_time_lst[1].append(float(t[1]))

        log_lst=glob.glob(self.folder_name+'/*rank*.log')
        size=len(log_lst)
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

    def Get_hit_stats_single(self,rank):
        log_file=glob.glob(self.folder_name+'/*-rank%d.log'%(rank))[0]
        fl=open(log_file,'r')
        txt=fl.readlines()
        fl.close()
        Eiger_file_list_file=txt[0][:-1].split(': ',1)[1]
        Eiger_file_run_lst=[]
        Hit_count=[]
        Eve_count=[]
        Hit_rate=[]
        for l in range(len(txt)):
            ll=txt[l]
            t=re.findall('data_.*\.h5',ll)

            if len(t)!=0:
                Eiger_file=ll[:-1]
                Eiger_file_short=os.path.basename(Eiger_file)
                Eiger_file_pre=Eiger_file_short.split('_data_',1)[0]
                Eiger_file_run=int(Eiger_file_short.split('_data_',1)[1].split('.',1)[0])
                Eiger_file_run_lst.append(Eiger_file_run)
                if (l+1)<len(txt):
                    Hit_count.append(re.findall('\d+',txt[l+1])[0])
                    Eve_count.append(re.findall('\d+',txt[l+1])[1])
                else:
                    Hit_count.append(0)
                    Eve_count.append(0)
                if (l+2)<len(txt):
                    Hit_rate.append(re.findall('\d+\.\d+',txt[l+2])[0])
                else:
                    Hit_rate.append(0)

        ind=np.argsort(Eiger_file_run_lst)
        Eiger_file_run_lst=np.array(Eiger_file_run_lst)[ind]
        Hit_count=np.array(Hit_count)[ind]
        Eve_count=np.array(Eve_count)[ind]
        Hit_rate=np.array(Hit_rate)[ind]
        return {'Eiger_file_list_file':Eiger_file_list_file,'Eiger_file_pre':Eiger_file_pre,'Eiger_file_run_lst':Eiger_file_run_lst,\
                'Hit_count':Hit_count,'Eve_count':Eve_count,'Hit_rate':Hit_rate}
    def Get_peak_raw_stats_single(self,rank):
        pk_file=glob.glob(self.folder_name+'/*-rank%d.pk'%(rank))[0]
        peak_arry=np.genfromtxt(pk_file,skip_header=1)
        peak_arry=peak_arry[:,1:]
        return peak_arry

    def Get_peak_raw_stats_all(self):
        size=self.comp_time_lst['num_workers']
        Eiger_file_pre=self.Get_hit_stats_single(0)['Eiger_file_pre']
        peak_arry_all=np.array([]).reshape(-1,7)
        for rank in range(size):
            peak_arry=self.Get_peak_raw_stats_single(rank)
            peak_arry_all=np.concatenate((peak_arry_all,peak_arry),axis=0)
        return Eiger_file_pre,peak_arry_all

    def Get_peak_stats_all(self,cam_len=0.2,photon_energy=12000,pixel_size=100e-6,center_x=2185,center_y=2075):
        Eiger_file_pre,peak_arry_all=self.Get_peak_raw_stats_all()
        RD=np.sqrt((peak_arry_all[:,-3]-center_x)**2+(peak_arry_all[:,-2]-center_y)**2)
        scat_ang=np.rad2deg(np.arctan(RD*pixel_size/cam_len))
        wave_len=12398/photon_energy
        Res=(wave_len)/(2*np.sin(np.deg2rad(scat_ang)/2)) #in Angstrom
        peak_arry_new=np.concatenate((peak_arry_all,RD.reshape(-1,1),scat_ang.reshape(-1,1),Res.reshape(-1,1)),axis=-1)
        return Eiger_file_pre, peak_arry_new

    def Get_hit_stats_all(self):
        '''
        live mode: when the hit finding is still running.
        '''
        Eiger_file_list_file=(self.Get_hit_stats_single(0))['Eiger_file_list_file']
        f=open(Eiger_file_list_file,'r')
        Eiger_file_lst=f.readlines()
        f.close()
        num_Eiger_file=len(Eiger_file_lst)

        size=self.comp_time_lst['num_workers']
        Hit_count_all=np.ones((num_Eiger_file,size))*(-10)
        Hit_rate_all=np.ones((num_Eiger_file,size))*(-10)

        for rank in range(size):
            single_dict=self.Get_hit_stats_single(rank)
            Hit_count_ary=np.array(single_dict['Hit_count']).reshape(-1,)
            Hit_count_all[0:Hit_count_ary.shape[0],rank]=Hit_count_ary
            Hit_rate_ary=np.array(single_dict['Hit_rate']).reshape(-1,)
            Hit_rate_all[0:Hit_rate_ary.shape[0],rank]=Hit_rate_ary

        Hit_rate_all_flat=Hit_rate_all.reshape(-1,)
        Hit_count_all_flat=Hit_count_all.reshape(-1,)
        return num_Eiger_file,Hit_count_all_flat,Hit_rate_all_flat





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


    Eiger_file_run_lst=np.array(vars()[short_name+'_obj'].Get_hit_stats_single(0)['Eiger_file_run_lst'])
    Eiger_file_pre=vars()[short_name+'_obj'].Get_hit_stats_single(0)['Eiger_file_pre']

    num_Eiger_file,Hit_count_all_flat,Hit_rate_all_flat=vars()[short_name+'_obj'].Get_hit_stats_all()

    #print(vars()[short_name+'_obj'].Get_hit_stats_single(0)['Hit_count'])
    #print(vars()[short_name+'_obj'].Get_hit_stats_single(0)['Eiger_file_run_lst'])




    plt.figure(num=1,figsize=(10,20))
    plt.subplot(2,1,1)
    plt.title('NSLS_hit_finding computation time plot')
    plt.plot(num_cores,mean_time,'r+',num_cores,max_time,'gs',num_cores,min_time,'b^')
    plt.xlabel('number of cores used for computation')
    plt.ylabel('computation time of all cores')
    leg_label=['mean_time','max_time','min_time']
    plt.legend(leg_label,loc='upper right')
    plt.xticks(np.arange(0,110,step=10))
    plt.subplot(2,1,2)
    plt.plot(num_cores,np.array(mean_time)/(num_Eiger_file*500),'r+',num_cores,np.array(max_time)/(num_Eiger_file*500),'gs',num_cores,np.array(min_time)/(num_Eiger_file*500),'b^')
    plt.xlabel('number of cores used for computation')
    plt.ylabel('computation time per pattern')
    leg_label=['mean_time','max_time','min_time']
    plt.legend(leg_label,loc='upper right')
    plt.xticks(np.arange(0,20,step=1))
    plt.savefig('comp_time.png')

    plt.figure(num=2)
    plt.plot(Hit_rate_all_flat,'bs',markersize=1)
    plt.xlabel('chucks of Eiger events in the *.lst file, in time series')
    plt.ylabel('Hit_rate %')
    plt.title(Eiger_file_pre)
    plt.savefig('Hit_rate.png')
    plt.show()

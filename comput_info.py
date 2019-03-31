'''
comput_info.py comstructs the Obeject Orientation framework to extract the
computation information.
'''

import sys,os
import numpy as np
import matplotlib.pyplot as plt
import glob

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

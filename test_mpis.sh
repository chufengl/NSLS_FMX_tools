#!/bin/bash


#SBATCH --job-name=HF_CFL
#SBATCH -p parallel
##SBATCH --mem-per-cpu=8000
#SBATCH -n 30
##SBATCH -N 1
#SBATCH -t 2-12:00
##SBATCH --mem-per-cpu=2000    
##SBATCH --array=20%8

##SBATCH -A chufengl
#SBATCH -o HF_%j.out
#SBATCH -e HF_%j.err
##SBATCH --mail-type=ALL
#SBATCH --mail-type=ALL        # notifications for job done & fail
#SBATCH --mail-user=chufengl@asu.edu # send-to address



export HDF5_PLUGIN_PATH=/bioxfel/software/h5plugin

mpiexec -n 30 python -u /home/chufengl/CFL_reposit/NSLS_FMX_tools/NSLS_FMX_utils_mpis.py ../../agave_test.lst 100 5 None 10 $1

#!/bin/bash
#SBATCH -J run_script_11
#SBATCH --partition=pi_esi
#SBATCH --time 120:00:00
#SBATCH --ntasks=12 --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH -o run_script_11.err

##################################################################
### Load the Gaussian module
module load Gaussian/16-C.01_AVX

##################################################################
### Set up Gaussian scracth directory

GAUSS_SCRDIR='/home/fas/batista/pev4/scratch60/gauss-scratch/'
export GAUSS_SCRDIR

##################################################################
### Run calculation

echo 'Starting program on'
date
echo ''

g16 < config_11_1.gjf > config_11_1.log
g16 < config_11_2.gjf > config_11_2.log
g16 < config_11_3.gjf > config_11_3.log
g16 < config_11_4.gjf > config_11_4.log
g16 < config_11_5.gjf > config_11_5.log
g16 < config_11_6.gjf > config_11_6.log
g16 < config_11_7.gjf > config_11_7.log
g16 < config_11_8.gjf > config_11_8.log
g16 < config_11_9.gjf > config_11_9.log
g16 < config_11_10.gjf > config_11_10.log
g16 < config_11_11.gjf > config_11_11.log
g16 < config_11_12.gjf > config_11_12.log
g16 < config_11_13.gjf > config_11_13.log
g16 < config_11_14.gjf > config_11_14.log
g16 < config_11_15.gjf > config_11_15.log
g16 < config_11_16.gjf > config_11_16.log
g16 < config_11_17.gjf > config_11_17.log
g16 < config_11_18.gjf > config_11_18.log
g16 < config_11_19.gjf > config_11_19.log
g16 < config_11_20.gjf > config_11_20.log
 
echo 'Finishing program on'
date
 

#!/bin/bash
#SBATCH -J run_script_9
#SBATCH --partition=pi_esi
#SBATCH --time 120:00:00
#SBATCH --ntasks=12 --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH -o run_script_9.err

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

g16 < config_9_1.gjf > config_9_1.log
g16 < config_9_2.gjf > config_9_2.log
g16 < config_9_3.gjf > config_9_3.log
g16 < config_9_4.gjf > config_9_4.log
g16 < config_9_5.gjf > config_9_5.log
g16 < config_9_6.gjf > config_9_6.log
g16 < config_9_7.gjf > config_9_7.log
g16 < config_9_8.gjf > config_9_8.log
g16 < config_9_9.gjf > config_9_9.log
g16 < config_9_10.gjf > config_9_10.log
g16 < config_9_11.gjf > config_9_11.log
g16 < config_9_12.gjf > config_9_12.log
g16 < config_9_13.gjf > config_9_13.log
g16 < config_9_14.gjf > config_9_14.log
g16 < config_9_15.gjf > config_9_15.log
g16 < config_9_16.gjf > config_9_16.log
g16 < config_9_17.gjf > config_9_17.log
g16 < config_9_18.gjf > config_9_18.log
g16 < config_9_19.gjf > config_9_19.log
g16 < config_9_20.gjf > config_9_20.log
 
echo 'Finishing program on'
date
 

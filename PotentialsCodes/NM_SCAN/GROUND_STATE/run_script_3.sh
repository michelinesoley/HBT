#!/bin/bash
#SBATCH -J run_script_3
#SBATCH --partition=pi_esi
#SBATCH --time 120:00:00
#SBATCH --ntasks=12 --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH -o run_script_3.err

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

g16 < config_3_1.gjf > config_3_1.log
g16 < config_3_2.gjf > config_3_2.log
g16 < config_3_3.gjf > config_3_3.log
g16 < config_3_4.gjf > config_3_4.log
g16 < config_3_5.gjf > config_3_5.log
g16 < config_3_6.gjf > config_3_6.log
g16 < config_3_7.gjf > config_3_7.log
g16 < config_3_8.gjf > config_3_8.log
g16 < config_3_9.gjf > config_3_9.log
g16 < config_3_10.gjf > config_3_10.log
g16 < config_3_11.gjf > config_3_11.log
g16 < config_3_12.gjf > config_3_12.log
g16 < config_3_13.gjf > config_3_13.log
g16 < config_3_14.gjf > config_3_14.log
g16 < config_3_15.gjf > config_3_15.log
g16 < config_3_16.gjf > config_3_16.log
g16 < config_3_17.gjf > config_3_17.log
g16 < config_3_18.gjf > config_3_18.log
g16 < config_3_19.gjf > config_3_19.log
g16 < config_3_20.gjf > config_3_20.log
 
echo 'Finishing program on'
date
 

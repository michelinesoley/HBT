#!/bin/bash
#SBATCH -J run_script_5
#SBATCH --partition=pi_esi
#SBATCH --time 120:00:00
#SBATCH --ntasks=12 --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH -o run_script_5.err

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

g16 < config_5_1.gjf > config_5_1.log
g16 < config_5_2.gjf > config_5_2.log
g16 < config_5_3.gjf > config_5_3.log
g16 < config_5_4.gjf > config_5_4.log
g16 < config_5_5.gjf > config_5_5.log
g16 < config_5_6.gjf > config_5_6.log
g16 < config_5_7.gjf > config_5_7.log
g16 < config_5_8.gjf > config_5_8.log
g16 < config_5_9.gjf > config_5_9.log
g16 < config_5_10.gjf > config_5_10.log
g16 < config_5_11.gjf > config_5_11.log
g16 < config_5_12.gjf > config_5_12.log
g16 < config_5_13.gjf > config_5_13.log
g16 < config_5_14.gjf > config_5_14.log
g16 < config_5_15.gjf > config_5_15.log
g16 < config_5_16.gjf > config_5_16.log
g16 < config_5_17.gjf > config_5_17.log
g16 < config_5_18.gjf > config_5_18.log
g16 < config_5_19.gjf > config_5_19.log
g16 < config_5_20.gjf > config_5_20.log
 
echo 'Finishing program on'
date
 

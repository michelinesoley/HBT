#!/bin/bash
#SBATCH -J run_script_8
#SBATCH --partition=pi_esi
#SBATCH --time 120:00:00
#SBATCH --ntasks=12 --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH -o run_script_8.err

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

g16 < config_8_1.gjf > config_8_1.log
g16 < config_8_2.gjf > config_8_2.log
g16 < config_8_3.gjf > config_8_3.log
g16 < config_8_4.gjf > config_8_4.log
g16 < config_8_5.gjf > config_8_5.log
g16 < config_8_6.gjf > config_8_6.log
g16 < config_8_7.gjf > config_8_7.log
g16 < config_8_8.gjf > config_8_8.log
g16 < config_8_9.gjf > config_8_9.log
g16 < config_8_10.gjf > config_8_10.log
g16 < config_8_11.gjf > config_8_11.log
g16 < config_8_12.gjf > config_8_12.log
g16 < config_8_13.gjf > config_8_13.log
g16 < config_8_14.gjf > config_8_14.log
g16 < config_8_15.gjf > config_8_15.log
g16 < config_8_16.gjf > config_8_16.log
g16 < config_8_17.gjf > config_8_17.log
g16 < config_8_18.gjf > config_8_18.log
g16 < config_8_19.gjf > config_8_19.log
g16 < config_8_20.gjf > config_8_20.log
 
echo 'Finishing program on'
date
 

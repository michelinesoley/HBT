#!/bin/bash
#SBATCH -J run_script_12
#SBATCH --partition=pi_esi
#SBATCH --time 120:00:00
#SBATCH --ntasks=12 --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH -o run_script_12.err

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

g16 < config_12_1.gjf > config_12_1.log
g16 < config_12_2.gjf > config_12_2.log
g16 < config_12_3.gjf > config_12_3.log
g16 < config_12_4.gjf > config_12_4.log
g16 < config_12_5.gjf > config_12_5.log
g16 < config_12_6.gjf > config_12_6.log
g16 < config_12_7.gjf > config_12_7.log
g16 < config_12_8.gjf > config_12_8.log
g16 < config_12_9.gjf > config_12_9.log
g16 < config_12_10.gjf > config_12_10.log
g16 < config_12_11.gjf > config_12_11.log
g16 < config_12_12.gjf > config_12_12.log
g16 < config_12_13.gjf > config_12_13.log
g16 < config_12_14.gjf > config_12_14.log
g16 < config_12_15.gjf > config_12_15.log
g16 < config_12_16.gjf > config_12_16.log
g16 < config_12_17.gjf > config_12_17.log
g16 < config_12_18.gjf > config_12_18.log
g16 < config_12_19.gjf > config_12_19.log
g16 < config_12_20.gjf > config_12_20.log
 
echo 'Finishing program on'
date
 

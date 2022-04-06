#!/bin/bash
#SBATCH -J run_script_1
#SBATCH --partition=pi_esi
#SBATCH --time 120:00:00
#SBATCH --ntasks=12 --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH -o run_script_1.err

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

g16 < config_1_1.gjf > config_1_1.log
g16 < config_1_2.gjf > config_1_2.log
g16 < config_1_3.gjf > config_1_3.log
g16 < config_1_4.gjf > config_1_4.log
g16 < config_1_5.gjf > config_1_5.log
g16 < config_1_6.gjf > config_1_6.log
g16 < config_1_7.gjf > config_1_7.log
g16 < config_1_8.gjf > config_1_8.log
g16 < config_1_9.gjf > config_1_9.log
g16 < config_1_10.gjf > config_1_10.log
g16 < config_1_11.gjf > config_1_11.log
g16 < config_1_12.gjf > config_1_12.log
g16 < config_1_13.gjf > config_1_13.log
g16 < config_1_14.gjf > config_1_14.log
g16 < config_1_15.gjf > config_1_15.log
g16 < config_1_16.gjf > config_1_16.log
g16 < config_1_17.gjf > config_1_17.log
g16 < config_1_18.gjf > config_1_18.log
g16 < config_1_19.gjf > config_1_19.log
g16 < config_1_20.gjf > config_1_20.log
 
echo 'Finishing program on'
date
 

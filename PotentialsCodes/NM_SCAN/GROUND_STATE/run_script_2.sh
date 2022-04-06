#!/bin/bash
#SBATCH -J run_script_2
#SBATCH --partition=pi_esi
#SBATCH --time 120:00:00
#SBATCH --ntasks=12 --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH -o run_script_2.err

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

g16 < config_2_1.gjf > config_2_1.log
g16 < config_2_2.gjf > config_2_2.log
g16 < config_2_3.gjf > config_2_3.log
g16 < config_2_4.gjf > config_2_4.log
g16 < config_2_5.gjf > config_2_5.log
g16 < config_2_6.gjf > config_2_6.log
g16 < config_2_7.gjf > config_2_7.log
g16 < config_2_8.gjf > config_2_8.log
g16 < config_2_9.gjf > config_2_9.log
g16 < config_2_10.gjf > config_2_10.log
g16 < config_2_11.gjf > config_2_11.log
g16 < config_2_12.gjf > config_2_12.log
g16 < config_2_13.gjf > config_2_13.log
g16 < config_2_14.gjf > config_2_14.log
g16 < config_2_15.gjf > config_2_15.log
g16 < config_2_16.gjf > config_2_16.log
g16 < config_2_17.gjf > config_2_17.log
g16 < config_2_18.gjf > config_2_18.log
g16 < config_2_19.gjf > config_2_19.log
g16 < config_2_20.gjf > config_2_20.log
 
echo 'Finishing program on'
date
 

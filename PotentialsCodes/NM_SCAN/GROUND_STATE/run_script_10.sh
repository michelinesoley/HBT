#!/bin/bash
#SBATCH -J run_script_10
#SBATCH --partition=pi_esi
#SBATCH --time 120:00:00
#SBATCH --ntasks=12 --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH -o run_script_10.err

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

g16 < config_10_1.gjf > config_10_1.log
g16 < config_10_2.gjf > config_10_2.log
g16 < config_10_3.gjf > config_10_3.log
g16 < config_10_4.gjf > config_10_4.log
g16 < config_10_5.gjf > config_10_5.log
g16 < config_10_6.gjf > config_10_6.log
g16 < config_10_7.gjf > config_10_7.log
g16 < config_10_8.gjf > config_10_8.log
g16 < config_10_9.gjf > config_10_9.log
g16 < config_10_10.gjf > config_10_10.log
g16 < config_10_11.gjf > config_10_11.log
g16 < config_10_12.gjf > config_10_12.log
g16 < config_10_13.gjf > config_10_13.log
g16 < config_10_14.gjf > config_10_14.log
g16 < config_10_15.gjf > config_10_15.log
g16 < config_10_16.gjf > config_10_16.log
g16 < config_10_17.gjf > config_10_17.log
g16 < config_10_18.gjf > config_10_18.log
g16 < config_10_19.gjf > config_10_19.log
g16 < config_10_20.gjf > config_10_20.log
 
echo 'Finishing program on'
date
 

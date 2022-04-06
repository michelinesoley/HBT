#!/bin/bash
#SBATCH -J run_script_4
#SBATCH --partition=pi_esi
#SBATCH --time 120:00:00
#SBATCH --ntasks=12 --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH -o run_script_4.err

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

g16 < config_4_1.gjf > config_4_1.log
g16 < config_4_2.gjf > config_4_2.log
g16 < config_4_3.gjf > config_4_3.log
g16 < config_4_4.gjf > config_4_4.log
g16 < config_4_5.gjf > config_4_5.log
g16 < config_4_6.gjf > config_4_6.log
g16 < config_4_7.gjf > config_4_7.log
g16 < config_4_8.gjf > config_4_8.log
g16 < config_4_9.gjf > config_4_9.log
g16 < config_4_10.gjf > config_4_10.log
g16 < config_4_11.gjf > config_4_11.log
g16 < config_4_12.gjf > config_4_12.log
g16 < config_4_13.gjf > config_4_13.log
g16 < config_4_14.gjf > config_4_14.log
g16 < config_4_15.gjf > config_4_15.log
g16 < config_4_16.gjf > config_4_16.log
g16 < config_4_17.gjf > config_4_17.log
g16 < config_4_18.gjf > config_4_18.log
g16 < config_4_19.gjf > config_4_19.log
g16 < config_4_20.gjf > config_4_20.log
 
echo 'Finishing program on'
date
 

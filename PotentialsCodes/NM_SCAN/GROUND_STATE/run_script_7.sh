#!/bin/bash
#SBATCH -J run_script_7
#SBATCH --partition=pi_esi
#SBATCH --time 120:00:00
#SBATCH --ntasks=12 --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH -o run_script_7.err

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

g16 < config_7_1.gjf > config_7_1.log
g16 < config_7_2.gjf > config_7_2.log
g16 < config_7_3.gjf > config_7_3.log
g16 < config_7_4.gjf > config_7_4.log
g16 < config_7_5.gjf > config_7_5.log
g16 < config_7_6.gjf > config_7_6.log
g16 < config_7_7.gjf > config_7_7.log
g16 < config_7_8.gjf > config_7_8.log
g16 < config_7_9.gjf > config_7_9.log
g16 < config_7_10.gjf > config_7_10.log
g16 < config_7_11.gjf > config_7_11.log
g16 < config_7_12.gjf > config_7_12.log
g16 < config_7_13.gjf > config_7_13.log
g16 < config_7_14.gjf > config_7_14.log
g16 < config_7_15.gjf > config_7_15.log
g16 < config_7_16.gjf > config_7_16.log
g16 < config_7_17.gjf > config_7_17.log
g16 < config_7_18.gjf > config_7_18.log
g16 < config_7_19.gjf > config_7_19.log
g16 < config_7_20.gjf > config_7_20.log
 
echo 'Finishing program on'
date
 

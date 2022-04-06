#!/bin/bash
#SBATCH -J run_script_6
#SBATCH --partition=pi_esi
#SBATCH --time 120:00:00
#SBATCH --ntasks=12 --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH -o run_script_6.err

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

g16 < config_6_1.gjf > config_6_1.log
g16 < config_6_2.gjf > config_6_2.log
g16 < config_6_3.gjf > config_6_3.log
g16 < config_6_4.gjf > config_6_4.log
g16 < config_6_5.gjf > config_6_5.log
g16 < config_6_6.gjf > config_6_6.log
g16 < config_6_7.gjf > config_6_7.log
g16 < config_6_8.gjf > config_6_8.log
g16 < config_6_9.gjf > config_6_9.log
g16 < config_6_10.gjf > config_6_10.log
g16 < config_6_11.gjf > config_6_11.log
g16 < config_6_12.gjf > config_6_12.log
g16 < config_6_13.gjf > config_6_13.log
g16 < config_6_14.gjf > config_6_14.log
g16 < config_6_15.gjf > config_6_15.log
g16 < config_6_16.gjf > config_6_16.log
g16 < config_6_17.gjf > config_6_17.log
g16 < config_6_18.gjf > config_6_18.log
g16 < config_6_19.gjf > config_6_19.log
g16 < config_6_20.gjf > config_6_20.log
 
echo 'Finishing program on'
date
 

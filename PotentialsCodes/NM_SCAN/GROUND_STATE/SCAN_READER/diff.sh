
path='../../../../DynamicsCodes/'
path='/Users/pablovidela/Dropbox/YALE/PROJECTS/PUMP_PROBE/UV-X_Ray/HBT/GAUSS_CALC/TDDFT/WB97XD/pvdz/carbontetrachloride/NORMAL_MODES/NM_SCAN/2D_SCAN/MODES_1_5/RELAXED/GROUND_STATE/SCAN_READER/'

for file in *.dat
do
	echo $file
#	diff $file ../../../../DynamicsCodes/ 
	diff $file $path






done

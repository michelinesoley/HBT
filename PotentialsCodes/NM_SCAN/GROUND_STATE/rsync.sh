folder=/gpfs/loomis/scratch60/fas/batista/pev4/HBT/GAUSS_CALC/TDDFT/WB97XD/pvdz/carbontetrachloride/NORMAL_MODES/NM_SCAN/2D_SCAN/MODES_1_5/GROUND_STATE/

#rsync -avz --dry-run --update --progress --exclude '*.chk' --exclude '*.sh' --exclude '*.err' --exclude 'fort.7' grace:$folder .
#rsync -avz --update --progress --exclude '*.chk' --exclude '*.sh' --exclude '*.err' --exclude 'fort.7' grace:$folder .

#cd SCAN_READER

# python 2d_scan_reader.py

#cd ..









  


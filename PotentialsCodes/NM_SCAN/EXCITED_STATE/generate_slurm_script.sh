###################################
### Generate slurm input files	###
###################################

rm run_script_*.sh

ndim1=15
ndim2=15

for i in `seq 1 $ndim1`;
do
  scriptname=run_script_$i.sh
  echo $scriptname

  ### copy header
  cp slurm_template.inp $scriptname

  for j in `seq 1 $ndim2`;
  do

    filename='config_'$i'_'$j
    
    echo $filename

    ### append run line
    echo "g16 < $filename.gjf > $filename.log" >> $scriptname

  done 

  echo " " >> $scriptname
  echo "echo 'Finishing program on'" >> $scriptname
  echo "date" >> $scriptname
  echo " " >> $scriptname

done

### change header

for file in run_script*
do

  echo $file

  prefix=$(basename $file .sh)

#  echo $prefix
  sed -i.bk "s/startFile/$prefix/g" "$prefix"".sh"

done

rm *.bk


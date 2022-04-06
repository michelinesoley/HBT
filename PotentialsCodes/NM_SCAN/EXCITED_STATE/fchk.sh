for file in *.chk
do
	echo $file
	formchk $file
done

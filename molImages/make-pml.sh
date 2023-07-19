direct=.

for file in *.xyz
do
	sed "s/INSERT/$file/g" script.pml > "$file"-script.pml
done
 

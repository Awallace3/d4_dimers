direct=.

for file in *.xyz
do
	sed "s/INSERT/$file/g" script.pml > "$file"-script.pml
	~cdsgroup/miniconda/envs/pymol3env/bin/pymol -cqr "$file"-script.pml
	rm "$file"-script.pml
done
 

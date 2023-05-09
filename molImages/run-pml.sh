direct=.

for file in *.pml
do
	~cdsgroup/miniconda/envs/pymol3env/bin/pymol -cqr "$file"	
done
 

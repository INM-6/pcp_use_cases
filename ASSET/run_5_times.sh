# Run an application 5 times and export to file

FILE_NAME="Jureca_16_ranks.txt"

mpirun -n 16 -perhost 16 python asset_min.py| tee $FILE_NAME
mpirun -n 16 -perhost 16 python asset_min.py | tee -a $FILE_NAME
mpirun -n 16 -perhost 16 python asset_min.py| tee -a $FILE_NAME
mpirun -n 16 -perhost 16 python asset_min.py | tee -a $FILE_NAME
mpirun -n 16 -perhost 16 python asset_min.py| tee -a $FILE_NAME
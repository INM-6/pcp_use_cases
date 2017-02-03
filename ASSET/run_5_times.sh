# Run an application 5 times and export to file

FILE_NAME="knl_mpi_gpp.txt"

mpirun -n 32 -perhost 32 python asset_min.py| tee $FILE_NAME
mpirun -n 32 -perhost 32 python asset_min.py | tee -a $FILE_NAME
mpirun -n 32 -perhost 32 python asset_min.py| tee -a $FILE_NAME
mpirun -n 32 -perhost 32 python asset_min.py | tee -a $FILE_NAME
mpirun -n 32 -perhost 32 python asset_min.py| tee -a $FILE_NAME
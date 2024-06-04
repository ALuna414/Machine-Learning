#!/bin/bash

#SBATCH --job-name=HW4-Part1
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --mem=5Gb
#SBATCH --time=00:10:00

echo ""
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

# Move to the correct directory

cd /gpfs/home/a_l523/HW4
echo "Current working directory is `pwd`"

python Part3.py


# end of the program
echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
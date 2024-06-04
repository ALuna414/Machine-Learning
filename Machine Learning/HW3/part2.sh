#!/bin/bash

#SBATCH --job-name=hw3p1
#SBATCH --partition=shared
#SBATCH -n 14
#SBATCH --time=00:10:00

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# go to my current WORKING directory
cd /gpfs/home/a_l523/HW3

python HW3/part2.py

echo "Program finished with exit code $? at: `date`"
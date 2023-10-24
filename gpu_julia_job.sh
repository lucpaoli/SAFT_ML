#!/bin/bash
#PBS -lselect=1:ncpus=2:mem=10gb:ngpus=1
#PBS -lwalltime=05:00:00
#PBS -N gpu_julia

# PBS_O_WORKDIR and TMPDIR are both loaded as env variables
# better to copy scripts to and from TMPDIR
cd ${PBS_O_WORKDIR}

# Load production environment first
module load tools/prod
# module load SciPy-bundle/2022.05-foss-2022a
# module load julia/1.6.4 

# run the program
# python myprog.py path/to/input.txt
./julia gpu_job.jl
#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=100gb
#PBS -lwalltime=30:00:00
#PBS -N no_pre

# PBS_O_WORKDIR and TMPDIR are both loaded as env variables
# better to copy scripts to and from TMPDIR
cd ${PBS_O_WORKDIR}

# Load production environment first
module load tools/prod

# run the program
# python myprog.py path/to/input.txt
../../../julia --threads 4 job.jl
#!/bin/bash
#PBS -lselect=1:ncpus=1:mem=64gb
#PBS -lwalltime=20:00:00
#PBS -N no_pre

# PBS_O_WORKDIR and TMPDIR are both loaded as env variables
# better to copy scripts to and from TMPDIR
cd ${PBS_O_WORKDIR}

# Load production environment first
module load tools/prod

# run the program
# python myprog.py path/to/input.txt
../../../julia --threads 1 job.jl
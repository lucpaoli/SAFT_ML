#!/bin/bash
#PBS -lselect=1:ncpus=20:mem=128gb
#PBS -lwalltime=20:00:00
#PBS -N fixed_maybe_1e-5

# PBS_O_WORKDIR and TMPDIR are both loaded as env variables
# better to copy scripts to and from TMPDIR
cd ${PBS_O_WORKDIR}

# Load production environment first
module load tools/prod

# run the program
# python myprog.py path/to/input.txt
../../../julia --threads 20 job.jl
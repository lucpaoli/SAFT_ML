#!/bin/bash
#PBS -lselect=1:ncpus=23:mem=64gb
#PBS -lwalltime=20:00:00
#PBS -N gnn_linear_attention

# PBS_O_WORKDIR and TMPDIR are both loaded as env variables
# better to copy scripts to and from TMPDIR
cd ${PBS_O_WORKDIR}

# Load production environment first
module load tools/prod
# module load SciPy-bundle/2022.05-foss-2022a
# module load julia/1.6.4 

# run the program
# python myprog.py path/to/input.txt
../../../julia --threads 23 job.jl
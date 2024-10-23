#!/bin/bash
#PBS -N train_ECG_bounded
#PBS -l select=1:ncpus=32:mem=120GB
#PBS -l walltime=40:00:00
#PBS -m bea
#PBS -M hugo.stackhouse@sydney.edu.au 
#PBS -V
cd "$PBS_O_WORKDIR"
julia --project=. --threads=auto instantiate.jl # ensure libraries according to manifest
julia --project=. --threads=auto --heap-size-hint=120G FinalBenchmarks/Interpolation/ECG200/interp_by_site.jl   # run your program, note the heap size limit

exit

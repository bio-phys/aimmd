#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./{jobname}.out.%j
#SBATCH -e ./{jobname}.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J {jobname}
#
# Queue (Partition):
#SBATCH --partition=s.phys  ## makes sure you run on the s. partition where left over resources of a node can be filled by other jobs
#SBATCH --gres=gpu:rtx6000:0  ## use only one GPU
#SBATCH --mem=4000  ## request less than a third (a third would be 64000) of the nodes available memory, this is important to leave over RAM for other jobs
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
###SBATCH --ntasks-per-core=2 ## uncommenting this enables hyperthreading, this means ranks*threads can be adjusted to up to 36. This most likely does not increase efficiency.
# Wall clock limit:
#SBATCH --time=24:00:00

# source the modules we use for this project
source /phys/ptmp/hejung/COV/source_modules.sh
# activate the correct python virtualenv (not really needed but to be complete)
#workon arcd_COV_distributed  # NOTE: workon does not work in scripts, use the line below!
source ~/.virtualenvs/aimmd_COV_distributed/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPI_NUM_RANKS=$SLURM_NTASKS_PER_NODE
export OMP_PLACES=cores  ## with enabled hyperthreading this line needs to be commented out
###export OMP_PLACES=threads ## with enabled hyperthreading these two lines need to be uncommented
###export SLURM_HINT=multithread

# Run gromacs with GPU offloading (most likely beneficial if you use 1GPU)
# some notes on the parameters:
# -pmefft gpu should not be needed (if -pme gpu is set this is the default), but it should also do no harm
# with -pme gpu we need to explicitly set -npme 1
# -update gpu i.e. doing nlist update on the GPU requires that the simulation uses a single rank only
# resethway in combination with maxh results in a reset of all performance counters after 7.5 min (i.e half the walltime), this should make the numbers more reliable as load balancing is then done (hopefully)
#srun gmx_mpi mdrun -v -s ../../bench.tpr -deffnm bench -cpi -noappend -pme gpu -pmefft gpu -update gpu -bonded gpu -nb gpu -ntomp $OMP_NUM_THREADS -resethway -maxh 0.25

srun {cmd_str}


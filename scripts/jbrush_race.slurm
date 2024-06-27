#!/bin/bash
#SBATCH --partition=gpu,day,week
#SBATCH --requeue
#SBATCH --job-name=brush_llm_race
#SBATCH --out="%x-%j.out"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --time=00-5:00:00
#SBATCH --mail-type=ALL

# do something
echo "I'm echoing to stdout"
echo "I'm echoing to stderr" 1>&2
echo "My JobID is ${SLURM_JOBID}"
echo "I have ${SLURM_CPUS_ON_NODE} CPUs on node $(hostname -s)"

module load miniconda
source activate langchain
cd /home/vs428/Documents/Uncertainty

date=$(date '+%Y-%m-%d-%H%M%S')

python brush_llm.py

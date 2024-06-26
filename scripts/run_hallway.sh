#!/bin/bash
#SBATCH --job-name=hallway         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=28        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=8:00:00         # total run time limit (HH:MM:SS)
#SBATCH --output=/home/jlidard/PredictiveRL/slurm/hallway_%j.out
#SBATCH --gres=gpu:1
##SBATCH --mail-type=begin        # send email when job begins
##SBATCH --mail-type=end          # send email when job ends
##SBATCH --mail-user=<YourNetID>@princeton.edu

module purge
module load anaconda3/2023.9
conda activate crl2

python /home/jlidard/PredictiveRL/scripts/run_hallway.py
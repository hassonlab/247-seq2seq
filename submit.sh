#!/bin/bash
#SBATCH --job-name=MeNTAL
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.err
#SBATCH --nodes=1 #nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-07:59:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-user=hvgazula@umich.edu

module purge
module load anaconda3
# module load cudnn7 cudatoolkit/10.1
conda activate torch-env

# Run tasks in parallel
# ConvNet10_${lr}_wd${weight_decay}_vmf${vocab_min_freq}
# BrainClassifier_${lr}_128_512_4_8_wd${weight_decay}_dr${dropout}_vmf${vocab_min_freq}_noam
# --gres=gpu:8 -n 1 --exclusive
# echo "Start time:" `date`
for vocab_min_freq in 10; do
  for vocab_max_freq in 250; do
    for max_num_bins in 60; do
      for lr in 0.0001; do
        for weight_decay in 0.05; do
          for dropout in 0.05; do
            for model in "MeNTAL"; do
              python brain2en.py --subjects 625 \
                                --max-electrodes 55 \
                                --model ${model} \
                                --lr ${lr} \
                                --tf-dropout ${dropout} \
                                --weight-decay ${weight_decay} \
                                --vocab-min-freq ${vocab_min_freq} \
                                --vocab-min-freq ${vocab_max_freq} &
            done
          done
        done;
      done;
    done;
  done;
done;
wait;
# echo "End time:" `date`

# Finish script
exit 0

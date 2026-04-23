#!/bin/bash
#PBS -N doom_train
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -o doom_train.log
cd $PBS_O_WORKDIR

# Set up conda environment
export PATH=/home/soft/anaconda3/bin:$PATH
source /home/soft/anaconda3/etc/profile.d/conda.sh

# Create and activate environment if needed
conda env list | grep -q doom-env || conda create -n doom-env python=3.9 -y
conda activate doom-env

# Install required packages
conda install -y pandas numpy scikit-learn pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
pip install transformers vaderSentiment requests google-api-python-client

# Run training
python train_model_full.py

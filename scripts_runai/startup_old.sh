#!/bin/bash
echo $USER
echo $HOME
echo $PWD

export PATH=/${HOME}/miniconda3/bin/:$PATH
cd /nfs/${HOME}/SACROS/ || exit

# set PYTORCH_CUDA_ALLOC_CONF to reduce fragmentation -- if needed
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Pass the arguments to the Python script
python run_monai_bundle.py --bundle "${1}" --mode "${2}" --fold "${3}"

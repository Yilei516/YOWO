#!/bin/bash

PORT=$1
CPUs=${2:-4}
GPUs=${3:-1}
MEM=${4:-'8G'}

if [ -z "$PORT" ]; then
    echo "Usage: bash $0 <PORT> [CPUs, GPUs, Mem]"
    exit
fi

echo
echo "This script will launch a slrum job that contains $CPUs CPUs, $GPUs  GPUs and $MEM memory on port $PORT"
echo "Using python from $(which python)"

echo """#!/bin/bash
#SBATCH -p rtx6000
#SBATCH --qos normal
#SBATCH --gres=gpu:$GPUs
#SBATCH -c $GPUs
#SBATCH --mem=$MEM
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter_notebook_info_$PORT.log
#SBATCH --ntasks=1

echo Running on \$(hostname)
date

jupyter notebook --ip 0.0.0.0 --port $PORT

""" > jupyter_at_$PORT.slrm

sbatch jupyter_at_$PORT.slrm

sleep 1
echo
echo "---------------------------- INSTRUCTIONS ---------------------------"
echo -e "\t1. Check the first line in jupyter_notebook_info_$PORT.log  to find out the node the job is running on"
echo -e "\t2. Check the output from the notebook to find the token for the website link"

echo
echo "Run the following command on your local machine"

echo
echo "ssh $USER@v.vectorinstitute.ai -NL $PORT:<the node name in step 1>:$PORT"

echo
echo "Navigate to your local browser, goto localhost:$PORT/?token=<the token in step 2>"


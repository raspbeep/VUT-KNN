#!/bin/bash
#PBS -N train_CycleGAN
#PBS -l walltime=12:00:00
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=16gb
#PBS -m ae
#PBS -j oe

# -m ae ...  send mail when the job terminates
# -j oe ... stderr stream of the job is merged with the stdout stream

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
# DATADIR must contain a folder with dataset:
# data/
#   train/
#     class1/
#     class2/
#   val/
#     class1/
#     class2/
HOMEDIR=/storage/praha1/home/$PBS_O_LOGNAME
DATADIR=$HOMEDIR/data/

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $HOMEDIR/jobs_info.txt

# loads the python modules
export PYTHONUSERBASE=/storage/brno12-cerit/home/xkrato61/pip3_libs

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# copy input data files to scratch directory
# if the copy operation fails, issue error message and exit
cp -r $DATADIR  $SCRATCHDIR || { echo >&2 "Error while copying data file(s)!"; exit 2; }

# navigate into scratch directory
cd $SCRATCHDIR

# create saved_images directory
mkdir saved_images

# clone repository with the source code 
git clone https://github.com/raspbeep/VUT-KNN.git

# navigate to code directory
cd VUT-KNN/

# checkout metacentrum_branch (for lower number of epochs)
git checkout metacentrum_test

# run training
python3 train.py --save $SCRATCHDIR/saved_images --data $SCRATCHDIR/data --epochs 2 > $HOMEDIR/training.out || { echo >&2 "Training ended up erroneously (with a code $?) !!"; exit 3; }

# move saved images to the home directory
cd $SCRATCHDIR
mv saved_images/ $HOMEDIR/ || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

# clean the SCRATCH directory
clean_scratch

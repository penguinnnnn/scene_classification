#!/usr/bin/env sh
USER=huangrenze
now=$(date +"%Y%m%d_%H%M%S")
CORE_ROOT=/mnt/lustre/${USER}/core

PART=Pose3
JOB=huangrenze
logfile=log/train-${now}.log

############ TRAIN
######## 1 >> local run, one gpu
## -- intelmpi
#${CORE_ROOT}/build/tools/caffe train --solver=solver.prototxt 2>&1 | tee ${logfile} &
#${CORE_ROOT}/build/tools/caffe train --solver=solver.prototxt --gpu=1 2>&1 | tee ${logfile} &
## -- openmpi
#${CORE_ROOT}/build/tools/caffe train --solver=solver.prototxt 2>&1 | tee ${logfile} &
## -- mvapich
#MV2_ENABLE_AFFINITY=0 ${CORE_ROOT}/build/tools/caffe train --solver=solver.prototxt 2>&1 | tee ${logfile} &

######## 2 >> mpirun
## -- intelmpi
#/mnt/lustre/share/intelmpi/bin/mpirun -np 8 ${CORE_ROOT}/build/tools/caffe train --solver=solver.prototxt 2>&1 | tee ${logfile} &
## -- openmpi
#/mnt/lustre/share/openmpi/bin/mpirun -np 8 ${CORE_ROOT}/build/tools/caffe train --solver=solver.prototxt 2>&1 | tee ${logfile} &
## -- mvapich
#MV2_ENABLE_AFFINITY=0 /mnt/lustre/share/mvapich/bin/mpirun -np 8 ${CORE_ROOT}/build/tools/caffe train --solver=solver.prototxt 2>&1 | tee ${logfile} &

######## 3 >> srun
## intelmpi and openmpi
GLOG_vmodule=MemcachedClient=-1 OMP_NUM_THREADS=1 srun --partition=${PART} --mpi=pmi2 --job-name=${JOB} --kill-on-bad-exit=1 --gres=gpu:4 -n4 --tasks-per-node=4 ${CORE_ROOT}/build/tools/caffe train --solver=solver.prototxt --weights=resnet50.caffemodel 2>&1 | tee ${logfile} &
## -- mvapich
#GLOG_vmodule=MemcachedClient=-1 OMP_NUM_THREADS=1 MV2_ENABLE_AFFINITY=0 srun --partition={PART} --mpi=pmi2 --job-name=example --kill-on-bad-exit=1 --gres=gpu:8 -n16 --ntasks-per-node=8 ${CORE_ROOT}/build/tools/caffe train --solver=solver.prototxt 2>&1 | tee ${logfile} &

############ TEST
#GLOG_vmodule=MemcachedClient=-1 OMP_NUM_THREADS=1 srun --partition=${PART} --mpi=pmi2 --job-name=example --kill-on-bad-exit=1 --gres=gpu:8 -n16 --tasks-per-node=8 ${CORE_ROOT}/build/tools/caffe test --model=./resnet/resnet50_v2.prototxt --iterations=781 --memory=liveness --weights=resnet50_v2_iter_1000.caffemodel 2>&1 | tee ${logfile} &

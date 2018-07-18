#!/usr/bin/python
import os

print('Start testing caffemodels...\n')

loss = []
top1 = []
top3 = []

cmd = 'GLOG_vmodule=MemcachedClient=-1 OMP_NUM_THREADS=1 srun --partition=Pose3 --mpi=pmi2 --job-name=huangrenze --kill-on-bad-exit=1 --gres=gpu:4 -n4 --tasks-per-node=4 /mnt/lustre/huangrenze/core/build/tools/caffe test --model=test.prototxt --weights=model/resnet50_iter_{iteration}.caffemodel 2>&1 | tee ./tmp.log &'

for i in range(0, 100):
    iteration = (i + 1) * 100
    print 'Iteration: %d' % (iteration)
    cmd_i = cmd.replace('{iteration}', str(iteration))
    test = os.popen(cmd_i)
    result = test.read()

    loss.append(float(result[result.rfind('Loss') + 6 : result.find('\n', result.rfind('Loss'))]))
    print(loss[i])
    top1.append(float(result[result.rfind('top1') + 7 : result.find('\n', result.rfind('top1'))]))
    print(top1[i])
    top3.append(float(result[result.rfind('top3') + 7 : result.find('\n', result.rfind('top3'))]))
    print(top3[i])
    test.close()

    print('\n')

min_loss = 5.00
max_top1 = 0.01
max_top3 = 0.01

loss_i = 0
top1_i = 0
top3_i = 0

for i in range(0, 100):
    if loss[i] < min_loss:
        min_loss = loss[i]
	loss_i = i
    if top1[i] > max_top1:
        max_top1 = top1[i]
        top1_i = i
    if top3[i] > max_top3:
        max_top3 = top3[i]
        top3_i = i

print 'Min loss: %f, iter: %d' % (min_loss, (loss_i + 1) * 100)
print 'Max top1: %f, iter: %d' % (max_top1, (top1_i + 1) * 100)
print 'Max top3: %f, iter: %d' % (max_top3, (top3_i + 1) * 100)

for i in range(0, 100):
    with open('test_result.txt', 'a') as f:
        f.write(str(i + 1))
        f.write(' ')
        f.write(str(loss[i]))
        f.write(' ')
        f.write(str(top1[i]))
        f.write(' ')
        f.write(str(top3[i]))
        f.write('\n')

print('\n...Done')

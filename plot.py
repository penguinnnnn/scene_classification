#!/usr/bin/python
import matplotlib.pyplot as plt

with open('test_result.txt', 'r') as f:
    x = f.read()

x = x.splitlines()
for i in range(0, 80):
    x[i] = x[i].split(' ')

num = []
loss = []
top1 = []
top3 = []

for i in range(0, 80):
    num.append(int(x[i][0]))
    loss.append(float(x[i][1]))
    top1.append(float(x[i][2]))
    top3.append(float(x[i][3]))

plt.figure(1)
plt.plot(num, loss, 'b')

plt.figure(2)
plt.plot(num, top1, 'g')
plt.plot(num, top3, 'r')

plt.show()

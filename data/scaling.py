import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

normalize_runtime = False
stat_file = 1

def add_plot(ax, stat_file):


    runtime_pairs = [("runtime_knl2", 36),   # == stat_file 0
                     ("runtime_jureca",4)]


    file_name = os.path.join(os.getcwd(),"pcp_use_cases","data",
                            runtime_pairs[stat_file][0] + ".csv")

    fp = open(file_name, "r")

    correction  = runtime_pairs[stat_file][1]
    rt_per_core = defaultdict(list)


    for line in fp.readlines():
        data = line.split(",")
        rt_per_core[eval(data[0].strip())].append(eval(data[1].strip()) - correction)




    cores_possible = [1,2,4,8,16,24,32,48, 64, 128, 256]

    cores = []
    runtime = []

    for key in cores_possible:
        if not key in rt_per_core:
            continue

        runtime.append(np.mean(rt_per_core[key]))
        cores.append(key)




    if normalize_runtime:
        norm_runtime = [x / runtime[0] for x in runtime]
    else: 
        norm_runtime = runtime


    ideal = [norm_runtime[0]]
    for idx in range(1, len(norm_runtime)):
        print cores[idx], ideal[0] / cores[idx]
        ideal.append(ideal[0] / cores[idx])

    ax.plot(cores, norm_runtime, label="measured")
    ax.plot(cores, ideal, label="ideal")
    print norm_runtime

# Create a figure get the axis
f, ax = plt.subplots()   

for idx in range(2):
    add_plot(ax, idx)

ax.set_xscale('log', basex=2)
ax.set_xticklabels([0,1,2,4,8,16,32,64,128,256])

plt.xlabel("nr of MPI ranks")
plt.ylabel("normalized runtime")
plt.legend()
if stat_file == 0:
    plt.title("Strong scaling jpm calculations on KNL compute node with MPI")
else:
    plt.title("Strong scaling jpm calculations on Jureca compute node with MPI")

plt.show()



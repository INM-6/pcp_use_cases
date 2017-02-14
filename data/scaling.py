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

    ax.plot(cores, norm_runtime,  "-o", label="Measured runtime ", linewidth=4, markersize=10, zorder=2)
    ax.plot(cores, ideal, "--", label="ideal", linewidth=2, zorder=1)
    print norm_runtime

# Create a figure get the axis
f, axes_pair = plt.subplots(1, 2, figsize=(20, 9))


title_pair = [
    "KNL accelerator",
    "Intel Broadwell 28 cores"
    ]
for idx,ax in enumerate(axes_pair):
    add_plot(ax, idx)

    ax.tick_params(axis='both', which='major',                   
                   labelsize=15)

    ax.set_xscale('log', basex=2)
    ax.set_xticklabels([0,1,2,4,8,16,32,64,128,256])
    ax.set_title(title_pair[idx], fontsize=30)
    ax.set_xlabel("nr of MPI ranks", fontsize=25)
    if idx==0:
        ax.set_ylabel("Runtime (sec.)",fontsize=25)

    plt.legend()
#plt.suptitle('Strong scaling behavior optimized part of the ASSET code', fontsize=30)
plt.show()
f.savefig('scale_plot.png', dpi=300, bbox_inches='tight')


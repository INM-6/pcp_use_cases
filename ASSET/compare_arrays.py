import numpy as np
import matplotlib.pyplot as plt
import collections


fp = open("data/new_output.npy", "r")

original = np.load(fp).astype(dtype=np.float32)

fp = open("diffent_output.npy", "r")

different = np.load(fp).astype(dtype=np.float32)




difference = original - different
norm_difference = difference / ((original + different) / 2)
difference = norm_difference

data = collections.defaultdict(int)
min = 9999
max = -9999

for entry in difference.flatten():
    if entry > max:
        max = entry
    if entry < min:
        min =entry

    data[entry] += 1



for orig, run2, diff in zip(original.flatten(),
                            different.flatten(),
                            difference.flatten()):
    print orig, run2, diff

print data.values()
print min, max


plt.hist(data.keys(), weights=data.values(), bins=50)

plt.show()
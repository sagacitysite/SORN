import numpy as np
import os
import glob
import matplotlib.pyplot as plt

current = "2017-08-30_13-48-17"
path = os.getcwd() + "/backup/test_multi/" + current
datapath = path + "/data"
plotpath = path + "/plots"

if not os.path.exists(plotpath):
    os.mkdir(plotpath)

files = glob.glob(os.path.join(datapath, "transition_distances_*"))
train_steps_unsorted = np.empty(len(files))
distances_unsorted = np.empty((len(files), len(np.load(files[0]))))
for i in range(len(files)):
    train_steps_unsorted[i] = int(files[i].split('transition_distances_')[1].split('.')[0])
    distances_unsorted[i] = np.load(files[i])

print(train_steps_unsorted)
print(distances_unsorted)

train_steps = np.sort(train_steps_unsorted)
distances = distances_unsorted[np.argsort(train_steps_unsorted)]

plt.plot(train_steps, np.mean(distances, axis=1))
plt.xlabel('Training steps')
plt.ylabel('Squared distance to initial transition')
#plt.show()
plt.savefig(plotpath + '/distances.png')
plt.close()
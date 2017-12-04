import numpy as np
import matplotlib.pyplot as plt
import seaborn
plt.style.use('ggplot')
import sys


logs = sys.argv[1:]

for log in logs:
  rews = []
  l = np.genfromtxt(log) 
  name = log.split('.')[0]
  for ln in l:
    rews.append(ln[4])
  plt.plot(rews, label=name)
plt.legend()
plt.ylabel('reward')
plt.xlabel('epochs')
plt.show()
  

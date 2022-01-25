import numpy as np
import matplotlib.pyplot as plt

cr = [.019, .003, .0004, .00005]
dr = [.011, .002, .0005, .00009]

x = np.arange(1,5)
width = 0.25

fig, ax = plt.subplots(figsize=(5,2.5))
cr_rects = ax.bar(x - .5*width, cr, width, label='Campus Robot', color='steelblue')
dr_rects = ax.bar(x + .5*width, dr, width, label='Disaster Relief', color='indianred')

labels = ['1', '2', '3', '4']
ax.set_ylabel('Percent')
# plt.ylim((0.0,0.5))
ax.set_xlabel(r'$\delta$')
ax.set_xticks(x)
ax.legend(fontsize='small')
# ax.set_xtick_labels(labels)

fig.tight_layout()
plt.savefig('reachability.png', dpi=600)

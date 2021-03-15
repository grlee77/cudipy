import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams['font.size'] = 16
rcParams['axes.titleweight'] = 'bold'

labels = ['SyN\n(2mm iso)',
          'SyN\n(1mm iso)',
          'Median\nOtsu',
          'Gibbs',
          'Tissue\nSegmentation']
gpu_means = [0.61, 3.22, 0.52, 3.11, 2.13]
cpu_means = [9.37, 69.1, 10.6, 6.32, 93.3]

qgpu_means = [0.52, 2.46, 0.268, 1.45, 2.34]
qcpu_means = [8.66, 127.9, 12.3, 8.95, 514]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 5.8))
rects1 = ax.bar(x - width / 2, cpu_means, width, label='CPU')
rects2 = ax.bar(x + width / 2, gpu_means, width, label='GPU')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Duration (s)')
ax.set_title('Duration (CPU vs GPU)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0, 100.0])
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

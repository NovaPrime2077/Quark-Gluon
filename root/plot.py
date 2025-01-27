import matplotlib.pyplot as plt
import numpy as np
all_features = cross_data.reshape(-1, 4)
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
axs = axs.ravel()
plt.style.use('dark_background')
L = ["Transverse Momentum","Rapidity","Azimuthal Angle","PDG id"]
for i in range(4):
    plt.style.use('dark_background')
    axs[i].scatter(range(len(all_features[:10000])), all_features[:10000, i], 
                   color='cyan', alpha=0.5, s=5)
    axs[i].set_title(f'Plot for {L[i]}', color='white')
    axs[i].set_xlabel('Particle Index', color='white')
    axs[i].set_ylabel(f'{L[i]}', color='white')
    axs[i].tick_params(axis='x', colors='white')
    axs[i].tick_params(axis='y', colors='white')
plt.tight_layout()
plt.show()
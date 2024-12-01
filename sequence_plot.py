import numpy as np
import matplotlib.pyplot as plt

# load traces
V1= np.load("V_1.npy")
I1= np.load("I_1.npy")
V2= np.load("V_2.npy")
I2= np.load("I_2.npy")
lbd_V1= np.load("lambda_V_1.npy")
lbd_I1= np.load("lambda_I_1.npy")
lbd_V2= np.load("lambda_V_2.npy")
lbd_I2= np.load("lambda_I_2.npy")

xmax = np.argmax(V1)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 4.2),
                       gridspec_kw={
                           'width_ratios': [3, 3],
                           'height_ratios': [2, 4],
                           'wspace': 0.1,
                           'hspace': 0.1}, sharey="row", sharex="all")

# make an x-axis line at y=0
ax[0,0].plot(V1)
xlim = np.asarray(ax[0,0].get_xlim())*0.6
for i in range(2):
    ax[i,0].plot(xlim, [ 0, 0 ], 'k', lw=1)
    ax[i,1].plot(xlim, [ 0, 0 ], 'k', lw=1)
    
# plot traces
ax[0,0].plot(I1)
ax[1,0].plot(lbd_V1)
ax[1,0].plot(lbd_I1)
ax[0,1].plot(V2)
ax[0,1].plot(I2)
ax[1,1].plot(lbd_V2)
ax[1,1].plot(lbd_I2)

ylim1 = ax[0,0].get_ylim()
ylim2 = ax[1,0].get_ylim()

# plot spike times lines
symbols = [ '--', '--', '-' ]
for i in range(2):
    for k,j in enumerate([0, 10, xmax]):
        ax[0,i].plot([ j, j ], ylim1, symbols[k], lw=1)
        ax[1,i].plot([ j, j ], ylim2, symbols[k], lw=1)

ax[1,0].plot(lbd_I1-lbd_V1)
ax[1,1].plot(lbd_I2-lbd_V2)

ax[0,0].set_axis_off()
ax[0,1].set_axis_off()
ax[1,0].set_yticks([])
ax[1,1].set_yticks([])
ax[1,0].set_xticks([])
ax[1,1].set_xticks([])
for spine in ["top", "right", "left", "bottom" ]:
    ax[1,0].spines[spine].set_visible(False)
    ax[1,1].spines[spine].set_visible(False)

ax[0,0].set_xlim(xlim)
# plot the weight and delay updates
for i in [ 0, 10 ]:
#    ax[0,1].plot([ i, i ], [0, lbd_I1[i]], lw=3, color="C1", solid_capstyle="butt")
#    ax[1,1].plot([ i, i ], [0, lbd_I2[i]], lw=3, color="C1", solid_capstyle="butt")
    ax[1,0].plot([ i, i ], [0, lbd_I1[i]-lbd_V1[i]], lw=3, color="k", solid_capstyle="butt")
    ax[1,1].plot([ i, i ], [0, lbd_I2[i]-lbd_V2[i]], lw=3, color="k", solid_capstyle="butt")

#plt.tight_layout()

plt.savefig("sequence_explain.pdf")
plt.show()


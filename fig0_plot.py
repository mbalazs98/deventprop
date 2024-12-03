import numpy as np
import matplotlib.pyplot as plt

DELAY = True
iv_low = -0.4
iv_high = 4
line_style = {
    "rec": '-',
    "sent": '--'
    }

d_clr = {
    "i": "C0",
    "v": "C1",
    "lambda_i": "C2",
    "lambda_v": "C3"
    }

lw= {
    "rec": 2,
    "sent": 1.5,
    "max": 1
    }

alpha = {
    "rec": 0.7,
    "sent": 1,
    "max": 0.7
    }

if DELAY:
    extra = "_delay"
else:
    extra = ""
iv_range = (iv_high-iv_low)+ 1
spacer = 4
lambda_scale = 3e5

# write down by hand which spike is which
spike_ids = [ 0, 1, 2, 0, 1, 3, 4, 3, 4, 2, 3, 4, 2 ]
back_spike_ids = [ 2, 3, 4 ]
mxcolor = "k"

sco = 4 # spike color offset

d= {}
for pop in "hid","out":
    d[pop]= {}
    for var in "i", "v", "lambda_v", "lambda_i":
        d[pop][var]= np.load(f"fig0_{var}_{pop}{extra}.npy")

for k in range(d["hid"]["v"].shape[0]-1):
    fig = plt.figure(figsize= (10,6))
    ax = fig.gca()
    
    i= 0
    T = d[pop]["lambda_v"].shape[1]
    t= np.arange(T,0,-1)
    st = {}
    ast = []
    st["rec"]= {}
    st["sent"]= {}
    for pop in "hid", "out":
        num = d[pop]["v"].shape[2]
        st["rec"][pop]= [0]*num
        st["sent"][pop]= [0]*num
        for n in range(num):
            curr = d[pop]["i"][k,:,...,n].flatten()
            nst = np.where(np.abs(np.diff(curr)) > 0.2)[0]
            st["rec"][pop][n] = nst
            ast.extend(nst)
            volt = d[pop]["v"][k,:,...,n].flatten()
            nst= np.where(np.abs(np.diff(volt)) > 0.1)[0]
            st["sent"][pop][n] = nst
            ast.extend(nst)
    num = d["out"]["v"].shape[2]   
    mx= []
    for n in range(num):
        mx.append(np.argmax(d["out"]["v"][k,:,...,n]))
    
                
    # start plotting
    scnt = 0
    for i in range(8):
        off = (7-i)*iv_range + (3-i//2)*spacer
        ax.plot([ 0, T ], [ off, off ], '-', color='k', lw=1)
    for s in ast:
        ax.plot([s, s], [ iv_low, iv_low+8*iv_range+3*spacer ], '-', color=[0.7, 0.7, 0.7], lw= 0.5)
    
    for m in mx:
        ax.plot([m, m], [ iv_low, iv_low+8*iv_range+3*spacer ], ':', color=[0.7, 0.7, 0.7], lw= 0.7)

    # plot the lines for spike times    
    i = 0
    for pop in "hid","out":
        num = d[pop]["v"].shape[2]
        for n in range(num):
            for which in "rec", "sent":
                for s in st[which][pop][n]:
                    low = (7-i)*iv_range+ (3-i//2)*spacer+iv_low
                    high = (7-i)*iv_range+ (3-i//2)*spacer+iv_high
                    ax.plot([s, s], [ low, high ], line_style[which], color=f"C{sco+spike_ids[scnt]}", lw=lw[which], alpha=alpha[which])
                    scnt += 1
            i += 1

    # plot the lines for maxima   
    low = (7-i)*iv_range+ (3-i//2)*spacer+iv_low
    high = (7-i)*iv_range+ (3-i//2)*spacer+iv_high
    ax.plot([mx[0], mx[0]], [ low, high ], "-", color=mxcolor, lw=lw["max"], alpha=alpha["max"])
    i += 1
    low = (7-i)*iv_range+ (3-i//2)*spacer-iv_high 
    high = (7-i)*iv_range+ (3-i//2)*spacer-iv_low
    ax.plot([m, m], [ low, high ], "-", color=mxcolor, lw=lw["max"], alpha=alpha["max"])
    i += 1

    # plot the lines for lambda jumps
    scnt = 0
    for s in st["sent"]["hid"][0]:
        low = (7-i)*iv_range+ (3-i//2)*spacer-iv_high 
        high = (7-i)*iv_range+ (3-i//2)*spacer-iv_low
        ax.plot([s, s], [ low, high ], "-", color=f"C{sco+back_spike_ids[scnt]}", lw=lw["rec"], alpha=alpha["rec"])
        scnt += 1
    i += 1
    for s in st["sent"]["hid"][1]:
        low = (7-i)*iv_range+ (3-i//2)*spacer+iv_low
        high = (7-i)*iv_range+ (3-i//2)*spacer+iv_high
        ax.plot([s, s], [ low, high ], "-", color=f"C{sco+back_spike_ids[scnt]}", lw=lw["rec"], alpha=alpha["rec"])
        scnt += 1
        
    i = 0
    for pop in "hid", "out":
        num = d[pop]["v"].shape[2]
        for n in range(num):
            for var in "i", "v":
                off = (7-i)*iv_range + (3-i//2)*spacer
                plot_d = d[pop][var][k,:,...,n] + off
                ax.plot(plot_d,color= d_clr[var])
            i += 1

    for pop in "out", "hid":
        num = d[pop]["lambda_v"].shape[2]   
        for n in range(num):
            for var in "lambda_v", "lambda_i":
                off = (7-i)*iv_range + (3-i//2)*spacer
                plot_d = d[pop][var][k+1,:,n] * lambda_scale + off                
                ax.plot(t-1,plot_d,color= d_clr[var])
            i += 1

    plt.axis("off")
    #for j in range(4):
    #    for s in st[pop]:
    #        ax[j,1].plot([s, s], [ iv_low, iv_high ], '--', color='k', lw= 1)
    #    j += 1

plt.xlim([ 50, T-50 ])
plt.tight_layout()      
plt.savefig(f"fig0{extra}.pdf")
            
plt.show()

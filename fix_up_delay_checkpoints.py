from glob import glob
import numpy as np
import os
import sys


if len(sys.argv) < 2:
    raise RuntimeError("Please pass checkpoint directory as command line argument")

delay_checkpoints = glob(os.path.join(sys.argv[1], "*-d.npy"))
for f in delay_checkpoints:
    print(f)
    d = np.load(f)
    mask = (np.round(d).astype(int) == 62)
    d[mask] = 0.0
    if np.sum(mask) > 0:
        print(f"Zeroed {np.sum(mask)} 62s in {f}")
        np.save(f, d)


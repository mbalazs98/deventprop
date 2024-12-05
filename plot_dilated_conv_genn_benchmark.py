import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gs

from itertools import chain
from pandas import read_csv

data = read_csv("dilated_conv_ml_genn.csv", delimiter=",")


data = data.sort_values(by="Max delay [ms]", ascending=False)

data_512_hidden = data[data["Num hidden"] == 512]
data_256_hidden = data[data["Num hidden"] == 256]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))


# Plot memory
actor_512 = axes[0].plot(data_512_hidden["Max delay [ms]"], 
                         data_512_hidden["mlGeNN peak memory [MiB]"],
                         marker="o")
actor_256 = axes[0].plot(data_256_hidden["Max delay [ms]"], 
                         data_256_hidden["mlGeNN peak memory [MiB]"], 
                         marker="o")
axes[0].plot(data_512_hidden["Max delay [ms]"], 
             data_512_hidden["DC max memory allocated [MiB]"],
             marker="o", color=actor_512[0].get_color(), linestyle="--")
axes[0].plot(data_256_hidden["Max delay [ms]"], 
             data_256_hidden["DC max memory allocated [MiB]"],
             marker="o", color=actor_256[0].get_color(), linestyle="--")


axes[1].plot(data_512_hidden["Max delay [ms]"], data_512_hidden["mlGeNN epoch time [s]"], marker="o", color=actor_512[0].get_color())
axes[1].plot(data_256_hidden["Max delay [ms]"], data_256_hidden["mlGeNN epoch time [s]"], marker="o", color=actor_256[0].get_color())
axes[1].plot(data_512_hidden["Max delay [ms]"], data_512_hidden["DC epoch time [s]"], marker="o", color=actor_512[0].get_color(), linestyle="--")
axes[1].plot(data_256_hidden["Max delay [ms]"], data_256_hidden["DC epoch time [s]"], marker="o", color=actor_256[0].get_color(), linestyle="--")

axes[0].set_ylabel("GPU memory [MiB]")
axes[1].set_ylabel("Training time per epoch [s]")
#axes[0].set_ylim((0, 3000))
#axes[1].set_ylim((0, 4000))

axes[0].set_title("A", loc="left")
axes[1].set_title("B", loc="left")

for a in axes[:2]:
    a.set_xlabel("Max delay timesteps")
    a.grid(axis="y")
    a.grid(which='minor', alpha=0.3)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

line_legend = fig.legend([actor_256[0],  mlines.Line2D([],[], color="black"), actor_512[0],mlines.Line2D([],[], linestyle="--", color="black")], 
                         ["256 hidden neurons", "mlGeNN", "512 hidden neurons",  "Dilated Convolutions"], 
                         loc="lower center", ncol=2, columnspacing=1.0)


fig.tight_layout(pad=0, rect=[0.0, 0.2, 1.0, 1.0])
fig.savefig("dilated_conv_genn_benchmark.pdf")

plt.show()

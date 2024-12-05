import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gs

from itertools import chain
from pandas import read_csv

# Load data and sort
data = read_csv("dilated_conv_ml_genn.csv", delimiter=",")
data = data.sort_values(by="Max delay [ms]", ascending=False)

# Filter rows with valid mlGeNN and DC data
ml_genn_data = data[data["mlGeNN epoch time [s]"].notnull() & data["mlGeNN peak memory [MiB]"].notnull()]
dc_data = data[data["DC epoch time [s]"].notnull() & data["DC max memory allocated [MiB]"].notnull()]

# Split into 512 and 256 hidden neurons
ml_genn_data_512_hidden = ml_genn_data[ml_genn_data["Num hidden"] == 512]
ml_genn_data_256_hidden = ml_genn_data[ml_genn_data["Num hidden"] == 256]

dc_data_512_hidden = dc_data[dc_data["Num hidden"] == 512]
dc_data_256_hidden = dc_data[dc_data["Num hidden"] == 256]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))


# Plot memory
actor_512 = axes[0].plot(ml_genn_data_512_hidden["Max delay [ms]"], 
                         ml_genn_data_512_hidden["mlGeNN peak memory [MiB]"] / 1024,
                         marker="o")
actor_256 = axes[0].plot(ml_genn_data_256_hidden["Max delay [ms]"], 
                         ml_genn_data_256_hidden["mlGeNN peak memory [MiB]"] / 1024, 
                         marker="o")
axes[0].plot(dc_data_512_hidden["Max delay [ms]"], 
             dc_data_512_hidden["DC max memory allocated [MiB]"] / 1024,
             marker="o", color=actor_512[0].get_color(), linestyle="--")
axes[0].plot(dc_data_256_hidden["Max delay [ms]"], 
             dc_data_256_hidden["DC max memory allocated [MiB]"] / 1024,
             marker="o", color=actor_256[0].get_color(), linestyle="--")


axes[1].plot(ml_genn_data_512_hidden["Max delay [ms]"], ml_genn_data_512_hidden["mlGeNN epoch time [s]"], marker="o", color=actor_512[0].get_color())
axes[1].plot(ml_genn_data_256_hidden["Max delay [ms]"], ml_genn_data_256_hidden["mlGeNN epoch time [s]"], marker="o", color=actor_256[0].get_color())
axes[1].plot(dc_data_512_hidden["Max delay [ms]"], dc_data_512_hidden["DC epoch time [s]"], marker="o", color=actor_512[0].get_color(), linestyle="--")
axes[1].plot(dc_data_256_hidden["Max delay [ms]"], dc_data_256_hidden["DC epoch time [s]"], marker="o", color=actor_256[0].get_color(), linestyle="--")

axes[0].set_ylabel("GPU memory [GiB]")
axes[1].set_ylabel("Training time per epoch [s]")
axes[0].set_yticks([0, 4, 8, 12, 16])
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

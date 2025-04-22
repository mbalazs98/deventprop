import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_utils import load_data_frame

from pandas import NamedAgg

from plot_settings import column_width, double_column_width

loihi_data = {("SHD", True, 62.0): 90.9, ("SHD", False, 62.0): 88.0,
              ("SHD", True, None): 89.48763251, ("SSC", True, None): 62.49632028,
              ("SSC", True, 62.0): 67.8, ("SSC", False, 62.0): 69.8}


bar_group_params = ["MAX_DELAY_STEPS", "DB", "RECURRENT"]

df = load_data_frame(bar_group_params, path=".", load_test=True)

# Splice in Loihi accuracy
df["test_loihi_accuracy"] = df.apply(lambda r: loihi_data.get((r["DB"], r["RECURRENT"], 
                                                               None if np.isnan(r["MAX_DELAY_STEPS"]) else r["MAX_DELAY_STEPS"])), 
                                     axis="columns")   

# Replace dataset and feedforward columns with single name
df["NAME"] = df.apply(lambda r: f"{r['DB']}\n{'Recurrent' if r['RECURRENT'] else 'Feedforward'}", axis=1)   

df = df.drop(["RECURRENT", "DB"], axis="columns")

fig, axis = plt.subplots(figsize=(column_width, 1.75))

# Group by bar group params and aggregate across repeats
group_df = df.groupby(["MAX_DELAY_STEPS", "NAME"], as_index=False, dropna=False)
group_df = group_df.agg(mean_test_accuracy=NamedAgg(column="test_accuracy", aggfunc="mean"),
                        std_test_accuracy=NamedAgg(column="test_accuracy", aggfunc="std"),
                        mean_test_loihi_accuracy=NamedAgg(column="test_loihi_accuracy", aggfunc="mean"),
                        std_test_loihi_accuracy=NamedAgg(column="test_loihi_accuracy", aggfunc="std"))

# Find unique hidden sizes and their indices
xticks, xtick_index = np.unique(group_df["NAME"], return_inverse=True)
group_df["XTICK_INDEX"] = xtick_index

# Split data into delayed and non-delayed
no_delay_df = group_df[group_df["MAX_DELAY_STEPS"].isnull()]
delay_df = group_df[group_df["MAX_DELAY_STEPS"].notnull()]

genn_no_delay_actor = axis.bar(no_delay_df["XTICK_INDEX"] + 0.2, no_delay_df["mean_test_accuracy"],
                               yerr=no_delay_df["std_test_accuracy"], width=0.2)
genn_delay_actor = axis.bar(delay_df["XTICK_INDEX"] + 0.4, delay_df["mean_test_accuracy"],
                            yerr=delay_df["std_test_accuracy"], width=0.2)
loihi_delay_actor = axis.bar(delay_df["XTICK_INDEX"] + 0.6, delay_df["mean_test_loihi_accuracy"],
                             yerr=delay_df["std_test_loihi_accuracy"], width=0.2)
loihi_no_delay_actor = axis.bar(no_delay_df["XTICK_INDEX"] + 0.8, no_delay_df["mean_test_loihi_accuracy"],
                                yerr=no_delay_df["std_test_loihi_accuracy"], width=0.2)

sns.despine(ax=axis)
axis.xaxis.grid(False)
axis.set_yticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
axis.set_xticks(np.arange(len(xticks)) + 0.5)
axis.set_ylabel("Accuracy [%]")
axis.set_xticklabels(xticks)

fig.legend([genn_no_delay_actor, genn_delay_actor, loihi_no_delay_actor, loihi_delay_actor], 
           ["mlGeNN (no delay)", "mlGeNN (delay)", "Loihi (no delay)", "Loihi (delay)"],
           loc="lower center", ncol=2, frameon=False)
fig.tight_layout(pad=0, rect=[0.0, 0.225, 1.0, 1.0])

fig.savefig("accuracy.pdf")

plt.show()

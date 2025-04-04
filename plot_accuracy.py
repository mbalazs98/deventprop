import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_utils import load_data_frame

from pandas import NamedAgg

from plot_settings import column_width, double_column_width

#loihi_data = {("shd", 256): 83.74558304, ("shd", 512): 89.48763251,
#             ("shd", 1024): 85.37985866, ("ssc", 256): 58.91472868,
#              ("ssc", 512): 61.74075164, ("ssc", 1024): 62.49632028}

bar_group_params = ["MAX_DELAY_STEPS", "DB", "RECURRENT"]

df = load_data_frame(bar_group_params, path=".", load_test=True)

# Splice in Loihi accuracy
#df["test_loihi_accuracy"] = df.apply(lambda r: loihi_data.get((r["dataset"], r["num_hidden"])), axis="columns")   
   
fig, axis = plt.subplots(figsize=(column_width, 1.75))

# Group by bar group params and aggregate across repeats
group_df = df.groupby(bar_group_params, as_index=False, dropna=False)
group_df = group_df.agg(mean_test_accuracy=NamedAgg(column="test_accuracy", aggfunc="mean"),
                        std_test_accuracy=NamedAgg(column="test_accuracy", aggfunc="std"))
                        #mean_test_loihi_accuracy=NamedAgg(column="test_loihi_accuracy", aggfunc="mean"),
                        #std_test_loihi_accuracy=NamedAgg(column="test_loihi_accuracy", aggfunc="std"),
print(group_df)
assert False
# Find unique hidden sizes and their indices
xticks, xtick_index = np.unique(group_df["num_hidden"], return_inverse=True)

genn_test_actor = axis.bar(xtick_index + 0.2, group_df["mean_test_accuracy"],
                           yerr=group_df["std_test_accuracy"], width=0.2)
#loihi_test_actor = axis.bar(xtick_index + 0.4, group_df["mean_test_loihi_accuracy"],
#                            yerr=group_df["std_test_loihi_accuracy"], width=0.2)


sns.despine(ax=axis)
ax.xaxis.grid(False)
axis.set_yticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
axis.set_xticks(np.arange(len(xticks)) + 0.3)

axis.set_xticklabels(xticks)

"""    
axes[0].set_ylabel("Accuracy [%]")

fig.legend(actors, ["mlGeNN train", "mlGeNN test", "Lava fixed-point test", "Loihi test"],
           loc="lower center", ncol=2, frameon=False)
fig.tight_layout(pad=0, rect=[0.0, 0.225, 1.0, 1.0])

fig.savefig("accuracy.pdf")
"""
plt.show()

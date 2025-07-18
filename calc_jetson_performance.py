import os
import numpy as np

from pandas import NamedAgg

from data_utils import load_data_frame

dataset_num_examples = {"SHD": 2264, "SSC": 20382}
            
for b in [1, 64, 128]:
    print(f"BATCH SIZE {b}")
    # Load dataframe       
    keys = ["MAX_DELAY_STEPS", "DB", "RECURRENT", "BATCH_SIZE"]
    df = load_data_frame(keys, lambda p: p["BATCH_SIZE"] == b,
                         path=".",
                         load_test_perf=True)
                        #load_test=True, load_test_perf=True, load_jetson_power=True)

    # Sort
    df = df.sort_values(["MAX_DELAY_STEPS", "DB", "RECURRENT"])

    # Calculate new columns
    df["test_gpu_time"] = df["test_neuron_update_time"] + df["test_presynaptic_update_time"] + df["test_custom_update_reset_time"]
    """
    # Drop some columns we now don't care about
    df = df.drop(columns=["test_accuracy", "test_neuron_update_time", 
                          "test_presynaptic_update_time", "test_custom_update_reset_time",
                          "test_custom_update_gradient_batch_reduce_time", "test_custom_update_gradient_learn_time",
                          "test_custom_update_batch_softmax_1_time", "test_custom_update_batch_softmax_2_time",
                          "test_custom_update_batch_softmax_3_time", "test_custom_update_spike_count_reduce_time",
                          "test_custom_update_zero_out_post_time", "dataset"])
    """
    num_examples = df["DB"].map(dataset_num_examples)
    num_batches = np.ceil(num_examples / b).astype(int)
    df["latency"] = df["test_gpu_time"] / num_batches
    """
    total_energy = df["test_gpu_time"] * df["jetson_sim_power"]
    total_dynamic_energy = df["test_gpu_time"] * (df["jetson_sim_power"] - df["jetson_idle_power"])
    df["total_energy_per_example"] = total_energy / num_examples
    df["dynamic_energy_per_example"] = total_dynamic_energy / num_examples

    df["total_energy_per_example"] *= 1e3
    df["dynamic_energy_per_example"] *= 1e3
    """
    df["latency"] *= 1e3
    #df["total_energy_delay_product"] = df["total_energy_per_example"] * df["latency"]
    print(df)

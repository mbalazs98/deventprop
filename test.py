import csv
import numpy as np

from multiprocessing import Event, Process
from ml_genn.callbacks import Callback
from ml_genn.compilers import InferenceCompiler
from ml_genn.serialisers import Numpy

from glob import glob
from json import dump
from time import perf_counter, sleep

import os
import sys

from model import create_model
from data import Dataset
from arguments import (shd_arguments, shd_recurrent_arguments, ssc_arguments,
                       ssc_recurrent_arguments, yy_arguments)

KERNEL_PROFILING = True
RECORD_JETSON_POWER = False

class CSVTestLog(Callback):
    def __init__(self, filename, epoch, output_pop):
        # Create CSV writer
        self.file = open(filename, "w")
        self.csv_writer = csv.writer(self.file, delimiter=",")

        # Write header row
        self.csv_writer.writerow(["Epoch", "Num trials", "Number correct", "Time"])
        
        self.epoch = epoch
        self.output_pop = output_pop

    def on_test_begin(self):
        self.start_time = perf_counter()

    def on_test_end(self, metrics):
        m = metrics[self.output_pop]
        self.csv_writer.writerow([self.epoch, m.total, m.correct, 
                                  perf_counter() - self.start_time])
        self.file.flush()

def record_jetson_power_process(unique_suffix, stop_event):
    from jtop import jtop

    # Create CSV writer
    file = open(f"jetson_power_{unique_suffix}.csv", "w")
    csv_writer = csv.writer(file, delimiter=",")

    # Write header row
    csv_writer.writerow(["Time", "VDD_CPU_GPU_CV", "VDD_SOC"])

    with jtop() as jetson:
        # Loop at safe rate
        start_time = perf_counter()
        while jetson.ok():
            csv_writer.writerow([perf_counter() - start_time,
                                 jetson.power["rail"]["VDD_CPU_GPU_CV"]["power"] / 1000.0,
                                 jetson.power["rail"]["VDD_SOC"]["power"] / 1000.0,])

            if stop_event.wait(0):
                break

if len(sys.argv) < 2:
    raise RuntimeError("Please pass checkpoint directory as command line argument")

# Get set of epoch IDs there are checkpoints for
checkpoints = set(int(os.path.split(f)[1].split("-")[0]) 
                  for f in glob(os.path.join(sys.argv[1], "*.npy")))

if len(checkpoints) == 0:
    raise RuntimeError("No checkpoints found")

if len(checkpoints) > 1:
    raise RuntimeError("Checkpoints from multiple epochs found")


args = shd_recurrent_arguments

dataset = Dataset(args, True)
max_spikes, latest_spike_time = dataset.get_data_info()

serialiser = Numpy(sys.argv[1])
input, network, ff, rec, hidden, output = create_model(args, max_spikes)

max_example_timesteps = int(np.ceil(latest_spike_time / args.DT))

# Load network state from final checkpoint
checkpoints = tuple(checkpoints)
network.load(checkpoints, serialiser)

compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                             reset_in_syn_between_batches=True,
                             batch_size=args.BATCH_SIZE,
                             kernel_profiling=KERNEL_PROFILING)

# Figure out unique suffix for model data
unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items() if not arg.startswith("__"))

# Dump parameters to file
output_file_title = os.path.basename(os.path.normpath(sys.argv[1])) + f"_{args.BATCH_SIZE}"
with open(f"params_{output_file_title}.json", "w") as fp:
    dump({arg: val for arg, val in vars(args).items() 
          if not arg.startswith("__")}, fp)

if RECORD_JETSON_POWER:
    jetson_power_stop_event = Event()
    jetson_power_process = Process(target=record_jetson_power_process,
                                    args=(output_file_title, jetson_power_stop_event))
    jetson_power_process.start()

    # Sleep for 5 seconds to get idle power
    sleep(10)
    
model_name = (f"classifier_test_{md5(unique_suffix.encode()).hexdigest()}"
              if os.name == "nt" else f"classifier_test_{unique_suffix}")
compiled_net = compiler.compile(network, name=model_name)

with compiled_net:
    # Loop through epochs
    start_time = perf_counter()

    spikes_test, labels_test = dataset("test")
    
    callbacks = ["batch_progress_bar", 
                 CSVTestLog(f"test_output_{output_file_title}.csv",
                            checkpoints[0], output)]
                     
    # Evaluate model on numpy dataset
    start_time = perf_counter()
    metrics, _  = compiled_net.evaluate({input: spikes_test},
                                        {output: labels_test},
                                        callbacks=callbacks)
    end_time = perf_counter()
    print(f"Accuracy = {100 * metrics[output].result}%")
    print(f"Time = {end_time - start_time}s")
    
    if KERNEL_PROFILING:
        genn_model = compiled_net.genn_model
        data = {
            "neuron_update": genn_model.neuron_update_time,
            "presynaptic_update": genn_model.presynaptic_update_time,
            "custom_update_reset": genn_model.get_custom_update_time("Reset")}
        with open(f"test_kernel_profile_{output_file_title}.json", "w") as fp:
            dump(data, fp)

if RECORD_JETSON_POWER:
    # Sleep for 5 seconds to get idle power
    sleep(10)

    jetson_power_stop_event.set();
    jetson_power_process.join()
    
from ml_genn_netx import export
export(f"{output_file_title}.net", input, output, dt=args.DT)

import numpy as np

from ml_genn.compilers import InferenceCompiler
from ml_genn.serialisers import Numpy

from glob import glob
from time import perf_counter

import os
import sys

from model import create_model
from data import Dataset
from arguments import (shd_arguments, shd_recurrent_arguments, ssc_arguments,
                       ssc_recurrent_arguments, yy_arguments)


if len(sys.argv) < 2:
    raise RuntimeError("Please pass checkpoint directory as command line argument")

# Get set of epoch IDs there are checkpoints for
checkpoints = set(int(os.path.split(f)[1].split("-")[0]) 
                  for f in glob(os.path.join(sys.argv[1], "*.npy")))

if len(checkpoints) == 0:
    raise RuntimeError("No checkpoints found")

if len(checkpoints) > 1:
    raise RuntimeError("Checkpoints from multiple epochs found")


args = ssc_arguments

dataset = Dataset(args, True)
max_spikes, latest_spike_time = dataset.get_data_info()


serialiser = Numpy(sys.argv[1])
input, network, ff, rec, hidden, output = create_model(args, max_spikes)

max_example_timesteps = int(np.ceil(latest_spike_time / args.DT))

# Load network state from final checkpoint
network.load(tuple(checkpoints), serialiser)

compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                                reset_in_syn_between_batches=True,
                                batch_size=args.BATCH_SIZE)

# Figure out unique suffix for model data
unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items() if not arg.startswith("__"))
model_name = (f"classifier_test_{md5(unique_suffix.encode()).hexdigest()}"
              if os.name == "nt" else f"classifier_test_{unique_suffix}")
compiled_net = compiler.compile(network, name=model_name)

with compiled_net:
    # Loop through epochs
    start_time = perf_counter()

    spikes_test, labels_test = dataset("test")
    # Evaluate model on numpy dataset
    start_time = perf_counter()
    metrics, _  = compiled_net.evaluate({input: spikes_test},
                                        {output: labels_test})
    end_time = perf_counter()
    print(f"Accuracy = {100 * metrics[output].result}%")
    print(f"Time = {end_time - start_time}s")
    
    
from ml_genn_netx import export
export(f"{os.path.basename(os.path.normpath(sys.argv[1]))}.net", input, output, dt=args.DT)

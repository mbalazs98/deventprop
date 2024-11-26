import numpy as np

from ml_genn import Connection, Network, Population
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential

from time import perf_counter
from ml_genn.utils.data import preprocess_spikes

from ml_genn.compilers.event_prop_compiler import default_params
import copy


BATCH_SIZE = 2
NUM_EPOCHS = 200
DT = 1.0
TRAIN = True

spikes = []
ind = np.array([[0,1],[1,0],[0,1],[1,0]])
time = np.array([[0,10],[0,10],[0,10],[0,10]])
labels = np.array([0,1,0,1])
for t, i in zip(time, ind):
    spikes.append(preprocess_spikes(t, i, 2))

max_spikes = 4
latest_spike_time = 25
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

num_input = 2
num_output = 2

rng = np.random.default_rng(1234)

network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                       num_input)
    output = Population(LeakyIntegrate(tau_mem=10.0, readout="max_var"),
                        num_output)

    # Connections
    in_out_delays = np.zeros((num_input, num_output))
    #in_out_delays[0,1], in_out_delays[1,0] = 10, 10
    in_out = Connection(input, output, Dense(Normal(mean=1, sd=0), in_out_delays),
                        Exponential(5), max_delay_steps=15)

max_example_timesteps = int(np.ceil(latest_spike_time / DT))
compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                losses="sparse_categorical_crossentropy",
                                reg_lambda_upper=0, reg_lambda_lower=0, 
                                reg_nu_upper=1, max_spikes=4, 
                                delay_learn_conns=[in_out],
                                optimiser=Adam(0), delay_optimiser=Adam(1.0), batch_size=BATCH_SIZE)
compiled_net = compiler.compile(network)

with compiled_net:
                    
    for e in range(NUM_EPOCHS):
        metrics, cb  = compiled_net.train({input: spikes},
                                        {output: labels},
                                        start_epoch=e, num_epochs=1)
        if 100 * metrics[output].result == 100:
            break

    print(f"Accuracy = {100 * metrics[output].result}%")

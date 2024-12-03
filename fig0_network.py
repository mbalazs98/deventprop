import numpy as np

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder, ConnVarRecorder
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from isyn_recorder import IsynRecorder

from time import perf_counter
from ml_genn.utils.data import preprocess_spikes

from ml_genn.compilers.event_prop_compiler import default_params
import copy


BATCH_SIZE = 1
NUM_EPOCHS = 1
DT = 0.1
TRAIN = True
DELAY = True

spikes = []
ind = np.array([[0,1], [0,1]])
time = np.array([[10,20],[10,20]])
labels = np.array([0,0])

for t, i in zip(time, ind):
    spikes.append(preprocess_spikes(t, i, 2))

g_in_hid = np.asarray([[ 2.5, 4.1 ], [ 2.5, 3.1 ]])
g_hid_out = np.asarray([[ -0.6667, 3.3333 ], [ 3.3333, 1.6667 ]])
print(g_in_hid)
if DELAY:
    in_hid_delays = np.asarray([[ 50, 0], [ 50, 0 ]])
    hid_out_delays = np.asarray([[40, 50], [ 15, 25]])
else:
    in_hid_delays = np.asarray([[ 0, 0 ], [ 0, 0 ]])
    hid_out_delays = np.asarray([[ 0, 0 ], [ 0, 0 ]])

max_spikes = 20
latest_spike_time = 50
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

num_input = 2
num_hidden = 2
num_output = 2

rng = np.random.default_rng(1234)

network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                       num_input)
    hidden = Population(LeakyIntegrateFire(tau_mem=10.0),num_hidden)
    # output = Population(LeakyIntegrate(tau_mem=10.0, readout="avg_var_exp_weight"),
    #                     num_output)
    output = Population(LeakyIntegrate(tau_mem=10.0, readout="max_var"),
                        num_output)

    # Connections
    in_hid = Connection(input, hidden, Dense(g_in_hid, in_hid_delays),
                        Exponential(5), max_delay_steps=100)
    hid_out = Connection(hidden, output, Dense(g_hid_out, hid_out_delays),
                         Exponential(5), max_delay_steps=100)

#1000000000.0
max_example_timesteps = int(np.ceil(latest_spike_time / DT)) + 15
compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                             losses="sparse_categorical_crossentropy",
                             reg_lambda_upper=0, reg_lambda_lower=0, 
                             reg_nu_upper=14, max_spikes=20, 
                             optimiser=Adam(0), batch_size=BATCH_SIZE,
                             dt= DT)
compiled_net = compiler.compile(network)
#print(compiled_net.genn_model.neuron_populations['Pop1']._neuron_model.get_sim_code())

print(compiled_net.connection_populations)

with compiled_net:
    callbacks = [
        VarRecorder(hidden, genn_var="LambdaV", key="lambda_v_hid"),
        VarRecorder(hidden, genn_var="LambdaI", key="lambda_i_hid"),
        VarRecorder(hidden, genn_var="V", key="v_hid"),
        IsynRecorder(in_hid, key="i_hid"),
        VarRecorder(output, genn_var="LambdaV", key="lambda_v_out"),
        VarRecorder(output, genn_var="LambdaI", key="lambda_i_out"),
        VarRecorder(output, genn_var="V", key="v_out"),
        IsynRecorder(hid_out, key="i_out"),
               ]
                    
    ho = compiled_net.connection_populations[hid_out]
    ho.vars["g"].pull_from_device()
    print(ho.vars["g"].view[:])
    for e in range(NUM_EPOCHS):
        metrics, cb  = compiled_net.train({input: spikes},
                                        {output: labels},
                                        start_epoch=e, num_epochs=1,
                                        callbacks=callbacks)

        ho = compiled_net.connection_populations[hid_out]
        ho.vars["g"].pull_from_device()
        print(ho.vars["g"].view[:])
        if DELAY:
            extra="_delay"
        else:
            extra=""
        np.save(f"fig0_lambda_v_hid{extra}.npy", cb["lambda_v_hid"])
        np.save(f"fig0_lambda_i_hid{extra}.npy", cb["lambda_i_hid"])
        np.save(f"fig0_v_hid{extra}.npy", cb["v_hid"])
        np.save(f"fig0_i_hid{extra}.npy", cb["i_hid"])
        np.save(f"fig0_lambda_v_out{extra}.npy", cb["lambda_v_out"])
        np.save(f"fig0_lambda_i_out{extra}.npy", cb["lambda_i_out"])
        np.save(f"fig0_v_out{extra}.npy", cb["v_out"])
        np.save(f"fig0_i_out{extra}.npy", cb["i_out"])
        

import numpy as np

from hashlib import md5

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder, Callback
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense,FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from tonic import DiskCachedDataset
from tonic.datasets import SHD

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

from ml_genn.compilers.event_prop_compiler import default_params
import copy

from argparse import ArgumentParser

import os
from callbacks import CSVLog


parser = ArgumentParser()
parser.add_argument("--learn_ff", type=int, default=1,  help="Learn delays in feedforward connections")
parser.add_argument("--ff_init", type=int, default=300, help="Intialise delays in feedforward connection as this")
parser.add_argument("--learn_rec", type=int, default=0, help="Learn delays in recurrent connections")
parser.add_argument("--rec_init", type=int, default=0, help="Intialise delays in recurrent connection as this")
parser.add_argument("--delays_lr", type=float, default=0.1, help="Learning rate for the delays")
parser.add_argument("--seed", type=int, default=123, help="Random seed")
parser.add_argument("--k_reg", type=float, default=5e-11, help="Spike regularisation strength")
parser.add_argument("--num_hidden", type=int, default=256, help="Number of hidden neurons")
parser.add_argument("--speaker", type=int, default=0, help="Speaker to use as validation (4 and 5 are only in test)")
args = parser.parse_args()
learn_ff = bool(args.learn_ff)
learn_rec = bool(args.learn_rec)
print("learn_ff",learn_ff,"ff_init",args.ff_init,"learn_rec",learn_rec,"rec_init",args.rec_init, "k_reg",args.k_reg, "speaker",args.speaker)

NUM_HIDDEN = args.num_hidden
BATCH_SIZE = 256
NUM_EPOCHS = 200
EXAMPLE_TIME = 20.0
AUGMENT_SHIFT = 40
P_BLEND = 0.5
DT = 1.0

# Class implementing simple augmentation where all events
# in example are shifted in space by normally-distributed value
np.random.seed(args.seed)

# Figure out unique suffix for model data
unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items())

class EaseInSchedule(Callback):
    def __init__(self):
        pass
    def set_params(self, compiled_network, **kwargs):
        self._optimiser = compiled_network.optimisers[0][0]
    def on_batch_begin(self, batch):
        # Set parameter to return value of function
        if self._optimiser.alpha < 0.001 :
            self._optimiser.alpha = (self._optimiser.alpha) * (1.05 ** batch)
        else:
            self._optimiser.alpha = 0.001


class Shift:
    def __init__(self, f_shift, sensor_size):
        self.f_shift = f_shift
        self.sensor_size = sensor_size

    def __call__(self, events: np.ndarray) -> np.ndarray:
        # Shift events
        events_copy = copy.deepcopy(events)
        events_copy["x"] = events_copy["x"] + np.random.randint(-self.f_shift, self.f_shift)
        
        # Delete out of bound events
        events_copy = np.delete(
            events_copy,
            np.where(
                (events_copy["x"] < 0) | (events_copy["x"] >= self.sensor_size[0])))
        return events_copy

class Blend:
    def __init__(self, p_blend, sensor_size):
        self.p_blend = p_blend
        self.n_blend = 7644
        self.sensor_size = sensor_size
    def __call__(self, dataset: list, classes: list) -> list:
        
        for i in range(self.n_blend):
            idx = np.random.randint(0,len(dataset))
            idx2 = np.random.randint(0,len(classes[dataset[idx][1]]))
            assert dataset[idx][1] == dataset[classes[dataset[idx][1]][idx2]][1]
            dataset.append((self.blend(dataset[idx][0], dataset[classes[dataset[idx][1]][idx2]][0]), dataset[idx][1]))
            
        return dataset

    def blend(self, X1, X2):
        X1 = copy.deepcopy(X1)
        X2 = copy.deepcopy(X2)
        mx1 = np.mean(X1["x"])
        mx2 = np.mean(X2["x"])
        mt1 = np.mean(X1["t"])
        mt2 = np.mean(X2["t"])
        X1["x"]+= int((mx2-mx1)/2)
        X2["x"]+= int((mx1-mx2)/2)
        X1["t"]+= int((mt2-mt1)/2)
        X2["t"]+= int((mt1-mt2)/2)
        X1 = np.delete(
            X1,
            np.where(
                (X1["x"] < 0) | (X1["x"] >= self.sensor_size[0]) | (X1["t"] < 0) | (X1["t"] >= 1000000)))
        X2 = np.delete(
            X2,
            np.where(
                (X2["x"] < 0) | (X2["x"] >= self.sensor_size[0]) | (X2["t"] < 0) | (X2["t"] >= 1000000)))
        mask1 = np.random.rand(X1["x"].shape[0]) < self.p_blend
        mask2 = np.random.rand(X2["x"].shape[0]) < self.p_blend
        X1_X2 = np.concatenate([X1[mask1], X2[mask2]])
        idx= np.argsort(X1_X2["t"])
        X1_X2 = X1_X2[idx]
        return X1_X2




dataset = SHD(save_to="../data", train=True)
num_input = int(np.prod(dataset.sensor_size))
num_output = len(dataset.classes)
shift = Shift(AUGMENT_SHIFT, dataset.sensor_size)
blend = Blend(P_BLEND, dataset.sensor_size)

# Loop through dataset
max_spikes = 0
latest_spike_time = 0
raw_dataset = []
spikes_val = []
labels_val = []
classes = [[] for _ in range(20)]
for i, data in enumerate(dataset):
    events, label = data
    events = np.delete(events, np.where(events["t"] >= 1000000))
    # Add raw events and label to list
    if dataset.speaker[i] == args.speaker:
        
        spikes_val.append(preprocess_tonic_spikes(events, dataset.ordering,
                                              dataset.sensor_size))
        labels_val.append(label)
    else:
        classes[label].append(len(raw_dataset))
        raw_dataset.append((events, label))
        
        # Calculate max spikes and max times
        max_spikes = max(max_spikes, len(events))
        latest_spike_time = max(latest_spike_time, np.amax(events["t"]) / 1000.0)
 
max_spikes = max(max_spikes, calc_max_spikes(spikes_val))
latest_spike_time = max(latest_spike_time, calc_latest_spike_time(spikes_val))

dataset = SHD(save_to="../data", train=False)

spikes_test = []
labels_test = []
for i in range(len(dataset)):
    events, label = dataset[i]
    events = np.delete(events, np.where(events["t"] >= 1000000))
    spikes_test.append(preprocess_tonic_spikes(events, dataset.ordering,
                                            dataset.sensor_size))
    labels_test.append(label)

# Determine max spikes and latest spike time
max_spikes = max(max_spikes, calc_max_spikes(spikes_test))
latest_spike_time = max(latest_spike_time, calc_latest_spike_time(spikes_test))


#serialiser = Numpy("checkpoints_" + unique_suffix)
network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                       num_input)
    hidden = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=None),
                        NUM_HIDDEN, record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output)

    # Connections
    if not args.learn_ff and args.ff_init == 0:
        ff_init = 255
    else:
        ff_init = args.ff_init

    if not args.learn_rec and args.learn_rec == 0:
        rec_init = 255
    else:
        rec_init = args.rec_init
    Conn_Pop0_Pop1 = Connection(input, hidden, Dense(Normal(mean=0.03, sd=0.01), np.random.uniform(0, args.ff_init, size=(num_input, NUM_HIDDEN))),
               Exponential(5.0), max_delay_steps=args.learn_ff*1000+(1-args.learn_ff)*ff_init)
    Conn_Pop1_Pop1 = Connection(hidden, hidden, Dense(Normal(mean=0.0, sd=0.02), np.random.uniform(0, args.rec_init, size=(NUM_HIDDEN, NUM_HIDDEN))),
               Exponential(5.0), max_delay_steps=args.learn_rec*1000+(1-args.learn_rec)*rec_init)
    Conn_Pop1_Pop2 = Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))

max_example_timesteps = int(np.ceil(latest_spike_time / DT))
delay_learn_conns = []
if learn_ff:
    delay_learn_conns.append(Conn_Pop0_Pop1)
if learn_rec:
    delay_learn_conns.append(Conn_Pop1_Pop1)
compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                losses="sparse_categorical_crossentropy",
                                reg_lambda_upper=args.k_reg, reg_lambda_lower=args.k_reg, 
                                reg_nu_upper=14, max_spikes=1500,
                                delay_learn_conns=delay_learn_conns,
                                optimiser=Adam(0.001 * 0.01), delay_optimiser=Adam(args.delays_lr),
                                batch_size=BATCH_SIZE, rng_seed=args.seed)

model_name = (f"classifier_train_{md5(unique_suffix.encode()).hexdigest()}"
                  if os.name == "nt" else f"classifier_train_{unique_suffix}")
compiled_net = compiler.compile(network, name=model_name)

with compiled_net:
    # Loop through epochs
    start_time = perf_counter()
    callbacks = [CSVLog(f"results/train_output_{unique_suffix}.csv", output), SpikeRecorder(hidden, key="hidden_spikes", record_counts=True), EaseInSchedule()]
    validation_callbacks = [CSVLog(f"results/valid_output_{unique_suffix}.csv", output)]
    best_acc, best_e, best_acc_train = 0, 0, 0
    early_stop = 25
    for e in range(NUM_EPOCHS):
        # Apply augmentation to events and preprocess
        spikes_train = []
        labels_train = []
        blended_dataset = blend(copy.deepcopy(raw_dataset), classes)
        for events, label in blended_dataset:
            spikes_train.append(preprocess_tonic_spikes(shift(events), dataset.ordering,
                                                    dataset.sensor_size))
            labels_train.append(label)
        
        # Train epoch
        train_metrics, valid_metrics, train_cb, valid_cb  = compiled_net.train({input: spikes_train},
                                            {output: labels_train},
                                            start_epoch=e, num_epochs=1, 
                                            shuffle=True, callbacks=callbacks, validation_callbacks=validation_callbacks, validation_x={input: spikes_val}, validation_y={output: labels_val})

        
        
        hidden_spikes = np.zeros(NUM_HIDDEN)
        for cb_d in train_cb['hidden_spikes']:
            hidden_spikes += cb_d

        
        _Conn_Pop0_Pop1 = compiled_net.connection_populations[Conn_Pop0_Pop1]
        _Conn_Pop0_Pop1.vars["g"].pull_from_device()
        g_view = _Conn_Pop0_Pop1.vars["g"].view.reshape((num_input, NUM_HIDDEN))
        g_view[:,hidden_spikes==0] += 0.002
        _Conn_Pop0_Pop1.vars["g"].push_to_device()

        if np.count_nonzero(hidden_spikes==0) > NUM_HIDDEN/10:
            print(np.count_nonzero(hidden_spikes==0), " number of silent neurons in architecture", args.learn_ff, args.learn_rec, args.ff_init, args.rec_init, args.k_reg, "at epoch", e)

        #to make sure that delays are not drifting
        if learn_ff and not learn_rec:
            Conn_Pop0_Pop1_delay = compiled_net.connection_populations[Conn_Pop0_Pop1].vars["d"].view.reshape((num_input, NUM_HIDDEN))
            _Conn_Pop0_Pop1.vars["d"].pull_from_device()
            d_view = _Conn_Pop0_Pop1.vars["d"].view.reshape((num_input, NUM_HIDDEN))
            min_delay = np.min(d_view)
            if min_delay != 0:
                d_view -= min_delay
                _Conn_Pop0_Pop1.vars["d"].push_to_device()
        

        if valid_metrics[output].result > best_acc:
            best_acc = valid_metrics[output].result
            best_acc_train = train_metrics[output].result
            best_e = e
            early_stop = 25
        else:
            early_stop -= 1
            if early_stop < 0:
                break

    

import numpy as np

from hashlib import md5

from ml_genn.callbacks import Checkpoint, SpikeRecorder, Callback
from ml_genn.compilers import EventPropCompiler
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy

from time import perf_counter

from argparse import ArgumentParser

import os

from model import create_model
from data import Dataset
from arguments import shd_arguments, ssc_arguments, yy_arguments

args = ssc_arguments

np.random.seed(args.SEED)

# Figure out unique suffix for model data
unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items() if not arg.startswith("__"))

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

dataset = Dataset(args)
max_spikes, latest_spike_time = dataset.get_data_info()


serialiser = Numpy("checkpoints_" + unique_suffix)
input, network, ff, rec, hidden, output = create_model(args, max_spikes)

max_example_timesteps = int(np.ceil(latest_spike_time / args.DT))
delay_learn_conns = []
if args.LEARN_FF:
    for conn in ff:
        delay_learn_conns.append(conn)
if args.RECURRENT and args.LEARN_REC:
    for conn in rec:
        delay_learn_conns.append(conn)

k_reg = {}
for i, hid in enumerate(hidden):
    k_reg[hid] = args.K_REG[i]

if args.DB == "SHD" or args.DB =="SSC":
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                    losses="sparse_categorical_crossentropy",
                                    reg_lambda_upper=k_reg, reg_lambda_lower=k_reg, 
                                    reg_nu_upper=14, max_spikes=1500,
                                    delay_learn_conns=delay_learn_conns,
                                    optimiser=Adam(0.001 * 0.01), delay_optimiser=Adam(args.DELAYS_LR),
                                    batch_size=args.BATCH_SIZE, rng_seed=args.SEED)
else:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                losses="sparse_categorical_crossentropy",
                                optimiser=Adam(0.001 * 0.01), batch_size=args.BATCH_SIZE,
                                softmax_temperature=0.5, ttfs_alpha=0.1, dt=args.DT, delay_learn_conns=delay_learn_conns,
                                delay_optimiser=Adam(args.DELAYS_LR),
                                rng_seed=args.SEED)

model_name = (f"classifier_train_{md5(unique_suffix.encode()).hexdigest()}"
                  if os.name == "nt" else f"classifier_train_{unique_suffix}")
compiled_net = compiler.compile(network, name=model_name)

with compiled_net:
    # Loop through epochs
    start_time = perf_counter()
    callbacks = ["batch_progress_bar",EaseInSchedule(), Checkpoint(serialiser)]
    for i, hid in enumerate(hidden):
        callbacks.append(SpikeRecorder(hid, key="hidden_spikes_"+str(i), record_counts=True))
    if args.DB == "SSC":
        validation_callbacks = ["batch_progress_bar"]
    best_e, best_acc = 0, 0
    early_stop = 15
    for e in range(args.NUM_EPOCHS):
        spikes_train, labels_train = dataset("train")
        if args.DB == "SSC":
            spikes_valid, labels_valid = dataset("valid")            
            train_metrics, metrics, cb, valid_cb  = compiled_net.train({input: spikes_train},
                                                {output: labels_train},
                                                start_epoch=e, num_epochs=1, 
                                                shuffle=True, callbacks=callbacks, validation_callbacks=validation_callbacks, validation_x={input: spikes_valid}, validation_y={output: labels_valid})
        else:
            metrics, cb  = compiled_net.train({input: spikes_train},
                                                {output: labels_train},
                                                start_epoch=e, num_epochs=1, 
                                                shuffle=True, callbacks=callbacks)
        
        
        hidden_spikes = np.zeros(args.NUM_HIDDEN)
        for cb_d in cb['hidden_spikes_0']:
            hidden_spikes += cb_d

        Conn = compiled_net.connection_populations[ff[0]]
        Conn.vars["g"].pull_from_device()
        g_view = Conn.vars["g"].view.reshape((args.NUM_INPUT, args.NUM_HIDDEN))
        g_view[:,hidden_spikes==0] += 0.002
        Conn.vars["g"].push_to_device()

        for i in range(1,args.NUM_LAYER):
            hidden_spikes = np.zeros(args.NUM_HIDDEN)
            for cb_d in cb['hidden_spikes_'+str(i)]:
                hidden_spikes += cb_d

            Conn = compiled_net.connection_populations[ff[i]]
            Conn.vars["g"].pull_from_device()
            g_view = Conn.vars["g"].view.reshape((args.NUM_HIDDEN, args.NUM_HIDDEN))
            g_view[:,hidden_spikes==0] += 0.002
            Conn.vars["g"].push_to_device()
        

        if metrics[output].result > best_acc:
            best_acc = metrics[output].result
            best_e = e
            early_stop = 15
        else:
            early_stop -= 1
            if early_stop < 0:
                break

    

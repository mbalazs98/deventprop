from ml_genn import Connection, Network, Population
from ml_genn.connectivity import Dense
from ml_genn.neurons import LeakyIntegrateFire, SpikeInput, LeakyIntegrate
from ml_genn.synapses import Exponential
from ml_genn.compilers.event_prop_compiler import default_params
from ml_genn.initializers import Normal, Uniform

def create_model(args, max_spikes):
    network = Network(default_params)
    with network:
        # Populations
        input = Population(SpikeInput(max_spikes=args.BATCH_SIZE * max_spikes),
                        args.NUM_INPUT)
        hidden = [Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                            tau_refrac=None),
                            args.NUM_HIDDEN, record_spikes=True)]
        for i in range(args.NUM_LAYER-1):
            hidden.append(Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                            tau_refrac=None),
                            args.NUM_HIDDEN, record_spikes=True))
        if args.READOUT == "li":
            output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                                args.NUM_OUTPUT)
        else:
            output = Population(LeakyIntegrateFire(tau_mem=20.0, readout="first_spike_time"),
                        args.NUM_OUTPUT)

        # Connections
        ff = [Connection(input, hidden[0], Dense(Normal(mean=0.03, sd=0.01), Uniform(0, args.FF_INIT)),
                Exponential(5.0), max_delay_steps=1000)]
        if bool(args.RECURRENT):         
            rec = [Connection(hidden[0], hidden[0], Dense(Normal(mean=0.0, sd=0.02), Uniform(0, args.RECURRENT_INIT)),
                    Exponential(5.0), max_delay_steps=1000)]
        else:
            rec = [None]
        for i in range(args.NUM_LAYER-1):
            ff.append(Connection(hidden[i], hidden[i+1], Dense(Normal(mean=0.02, sd=0.03), Uniform(0, args.FF_INIT)),
                Exponential(5.0), max_delay_steps=1000))
            if bool(args.RECURRENT):
                rec.append(Connection(hidden[i+1], hidden[i+1], Dense(Normal(mean=0.0, sd=0.02), Uniform(0, args.RECURRENT_INIT)),
                    Exponential(5.0), max_delay_steps=1000))
            else:
                rec.append(None)
        Connection(hidden[-1], output, Dense(Normal(mean=0.0, sd=0.03)),
                Exponential(5.0))
    return network, ff, rec, hidden, output